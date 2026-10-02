#!/usr/bin/env bash
# Drive the JarvisLabs LLM run from an always-on Linux/macOS machine (or WSL), so nothing depends on a
# laptop staying connected. Needs only bash, ssh, scp, tar, tmux and curl; uv is installed if missing.
#
#   bash jarvis_drive.sh keygen                    once per machine: SSH key ~/.ssh/jarvis_ed25519; prints the public half
#   bash jarvis_drive.sh keys                      once: JarvisLabs API key (hidden input) -> ~/.config/clarity_jarvis/keys.env
#   bash jarvis_drive.sh connect "<ssh command>" <machine_id>
#                                                  the SSH command from the instance page, e.g.
#                                                  "ssh -p 11014 root@ssh.jarvislabs.net"; tests the connection
#   bash jarvis_drive.sh all                       push + start + watch (the overnight command)
#   bash jarvis_drive.sh create | resume           create a new VM (GPU_TYPE, 100 GB) or resume this profile's
#                                                  paused VM through the JarvisLabs API, then connect to it
#   Several VMs at once: prefix any command with JPROFILE=<name> (own host, machine id, logs, watcher),
#   e.g.  JPROFILE=b RUN_NAME=Q8_fullq_lora_12ep SEEDS="0 1" EXTRA_ARGS="--epochs 12" bash jarvis_drive.sh all
#
#   push     build jarvis_bundle.tgz here, copy it to the instance, unpack into /home/clarity_llm/code
#   start    run `jarvis_setup.sh all` on the instance, inside ITS tmux session (survives disconnects)
#   watch    local tmux session "jarvis-watch": every 5 min, copy the instance's logs and finished
#            outputs into clarity/logs/jarvis/ and clarity/runs/<NAME>/; when the pipeline says
#            ALL DONE or FAILED, copy logs, probabilities and metrics (LoRA adapters go to the HF repo from the instance, not here) and pause the instance
#   sync     one copy, now          status   the instance's progress          pause   pause it now
#   ssh      open a shell on the instance
#
# Pausing: this machine pauses the instance as soon as it has the results. The instance also
# holds the key and pauses itself PAUSE_DELAY (30 min) after finishing, as a backup in case
# this machine's watcher has died (results then stay on the paused instance's disk). Pausing stops GPU billing; storage keeps billing until you
# destroy the instance.
set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CFG=$HOME/.config/clarity_jarvis
KEYS=$CFG/keys.env
# One profile per VM, so several VMs can run side by side: JPROFILE=b bash jarvis_drive.sh ...
# Each profile has its own host + machine id, local log folder and watcher session.
PROFILE=${JPROFILE:-}
HOSTCFG=$CFG/host${PROFILE:+_$PROFILE}.env
GPU_TYPE=${GPU_TYPE:-RTX-PRO6000}
SSHKEY=$HOME/.ssh/jarvis_ed25519
NAME=${RUN_NAME:-Q8_fullq_lora}   # not NAME: WSL exports NAME=<hostname>
SEEDS=${SEEDS:-"0 1 2"}
EXTRA_ARGS=${EXTRA_ARGS:-}
MAX_TOTAL_HOURS=${MAX_TOTAL_HOURS:-5}
INTERVAL=${INTERVAL:-300}
RBASE=${RBASE:-/home/clarity_llm}
LOCAL_LOGS=$REPO/clarity/logs/jarvis${PROFILE:+/$PROFILE}
WATCH=jarvis-watch${PROFILE:+-$PROFILE}
LOCAL_RUNS=$REPO/clarity/runs
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

need_host() { load; [ -n "${JHOST:-}" ] || { echo "run: bash jarvis_drive.sh connect \"<ssh command>\" <machine_id>"; exit 2; }; }

pause_now() {
  load; need_uv
  [ -n "${JL_API_KEY:-}" ] && [ -n "${JL_MACHINE_ID:-}" ] || { note "cannot pause: no key or machine id"; return 1; }
  JL_API_KEY=$JL_API_KEY "$UVX" --quiet --from jarvislabs python -c \
    "from jarvislabs import Client; print('pause:', Client().instances.pause($JL_MACHINE_ID))" 2>&1 | tee -a "$LOCAL_LOGS/drive.log"
}

sync_once() {   # $1 = "final": same files, plus the packaged results tarball
  # Every run folder on the instance, without LoRA weights (adapter_best, epochs/ep*_adapter): those go
  # to the Hugging Face repo from the instance, and would be GBs in a local clone.
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
    || echo "key check failed -- re-run: bash jarvis_drive.sh keys"
  ;;

connect)
  cmd=${2:-}; mid=${3:-}
  [ -n "$cmd" ] && [ -n "$mid" ] || { echo 'usage: bash jarvis_drive.sh connect "ssh -p PORT USER@HOST" MACHINE_ID'; exit 2; }
  port=$(echo "$cmd" | sed -nE 's/.*-p[ ]*([0-9]+).*/\1/p'); port=${port:-22}
  dest=$(echo "$cmd" | grep -oE '[A-Za-z0-9._-]+@[A-Za-z0-9._-]+' | head -1)
  [ -n "$dest" ] || { echo "could not find user@host in: $cmd"; exit 2; }
  printf 'JUSER=%s\nJHOST=%s\nJPORT=%s\nJL_MACHINE_ID=%s\n' "${dest%@*}" "${dest#*@}" "$port" "$mid" > "$HOSTCFG"
  chmod 600 "$HOSTCFG"
  load
  note "connect: ${PROFILE:-default} -> $JUSER@$JHOST port $JPORT, machine $mid"
  ssh-keygen -R "$JHOST" >/dev/null 2>&1   # dynamic IPs are reused by other VMs: drop a stale host key
  for i in $(seq 1 18); do R true 2>/dev/null && break; sleep 10; done   # a new VM needs a minute for SSH
  R 'echo "connected to $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /home | tail -1' \
    || { echo "SSH failed. Is $SSHKEY.pub (bash jarvis_drive.sh keygen) added to JarvisLabs, and was the instance created after adding it?"; exit 1; }
  ;;

create|resume)
  # create: a new VM (GPU_TYPE, 100 GB) for this profile; resume: the profile's paused VM (new IP)
  load; need_uv
  [ -n "${JL_API_KEY:-}" ] || { echo "run: bash jarvis_drive.sh keys"; exit 2; }
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
  (cd "$REPO" && bash jarvis_setup.sh bundle >/dev/null) || { note "bundle failed"; exit 1; }
  # containers log in as root; VMs as a regular user (ubuntu) who needs sudo to create $RBASE under /home
  R "mkdir -p $RBASE/code 2>/dev/null || { sudo mkdir -p $RBASE && sudo chown \$(id -u):\$(id -g) $RBASE && mkdir -p $RBASE/code; }" \
    && scp -i "$SSHKEY" -P "$JPORT" -o StrictHostKeyChecking=accept-new -q "$REPO/jarvis_bundle.tgz" "$JUSER@$JHOST:$RBASE/jarvis_bundle.tgz" \
    && R "tar xzf $RBASE/jarvis_bundle.tgz -C $RBASE/code && ls $RBASE/code/clarity" \
    && note "push: bundle unpacked into $RBASE/code" || { note "push failed"; exit 1; }
  # W&B and Hugging Face keys travel separately from the bundle, owner-readable only
  if [ -f "$REPO/clarity/.env" ]; then
    R "umask 077; cat > $RBASE/code/clarity/.env" < "$REPO/clarity/.env" && note "push: clarity/.env copied (mode 600)"
  fi
  ;;

start)
  need_host
  # the instance gets the key too, only so it can pause itself if this server's watcher dies
  R "umask 077; printf 'JL_API_KEY=%s\nJL_MACHINE_ID=%s\n' '${JL_API_KEY:-}' '${JL_MACHINE_ID:-}' > $RBASE/keys.env"
  R "NAME='$NAME' SEEDS='$SEEDS' EXTRA_ARGS='$EXTRA_ARGS' MAX_TOTAL_HOURS=$MAX_TOTAL_HOURS AUTO_PAUSE=1 PAUSE_DELAY=1800 NO_PROMPT=1 \
     bash $RBASE/code/jarvis_setup.sh all" && note "start: pipeline launched on the instance (tmux clarity-llm)"
  ;;

watch)
  need_host
  if [ "${2:-}" != _inner ]; then
    tmux has-session -t "$WATCH" 2>/dev/null && { echo "already watching: tmux attach -t $WATCH"; exit 0; }
    tmux new-session -d -s "$WATCH" "JPROFILE='$PROFILE' RUN_NAME='$NAME' bash '$REPO/jarvis_drive.sh' watch _inner; exec bash"
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
          sync_once final && note "watch: final results copied to clarity/runs/ and $LOCAL_LOGS"
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
status) need_host; R "bash $RBASE/code/jarvis_setup.sh status" ;;
pause)  pause_now ;;
ssh)    need_host; ssh -i "$SSHKEY" -p "$JPORT" "$JUSER@$JHOST" ;;
*) sed -n '2,30p' "$0"; exit 2 ;;
esac
