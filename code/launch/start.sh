#!/usr/bin/env bash
# Start -- or, after a reboot or an ssh drop, resume -- an experiment on the lab server, inside tmux.
#
#   bash code/launch/start.sh E14     # every lane in code/experiments/E14/, one tmux window each
#
#   tmux attach -t clarity-E14        # watch;  Ctrl-b n / Ctrl-b p to switch lanes;  Ctrl-b d to detach
#   tail -f logs/E14.log              # one line per event, all lanes
#   tail -f logs/<run>_seed<k>.log    # one run's training output
#
# Safe to run twice: it refuses to start a second copy of an experiment that is running. Everything
# inside is resumable, so after a reboot this same command continues where the experiment stopped.
# To stop: `tmux kill-session -t =clarity-E14` ("=" = exact name; a bare prefix can match clarity-E14-watch) (never kill the process whose command line starts with
# `tmux new-session`: that is the tmux server, which hosts every session).
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
CODE="$(dirname "$DIR")"
EXP="${1:-}"
[ -n "$EXP" ] || { sed -n '2,13p' "$0"; exit 2; }

EXPDIR="$CODE/experiments/$EXP"
LANES=()   # lanes marked "# start: watcher" are started and stopped by gpu_watch.sh instead
for lane in "$EXPDIR"/lane_gpu*_*.txt; do grep -qE '^#[[:space:]]*start:[[:space:]]*watcher' "$lane" || LANES+=( "$lane" ); done
[ "${#LANES[@]}" -gt 0 ] || { echo "no lane files in $EXPDIR"; exit 2; }
if pgrep -f "^bash .*/run_queue\.sh .*/experiments/$EXP/" >/dev/null; then   # anchored: a `tmux new-session` client carries the same path and must not count
  echo "$EXP already running -- attach with: tmux attach -t clarity-$EXP"; exit 0
fi
SESSION="clarity-$EXP"
tmux kill-session -t "=$SESSION" 2>/dev/null   # "=": exact name (a prefix would match clarity-<EXP>-watch)
first=1
for lane in "${LANES[@]}"; do
  win="$(basename "$lane" .txt)"
  # tmux does not inherit these from the calling shell; pass them through if set
  envp=""; for v in CLARITY_RUNS CLARITY_PY WANDB_MODE HF_HOME QUEUE_PASS_SLEEP; do [ -n "${!v:-}" ] && envp+="$v=${!v} "; done
  cmd="env ${envp}bash $DIR/run_queue.sh $lane; echo LANE_EXITED; exec bash"
  if [ $first = 1 ]; then tmux new-session -d -s "$SESSION" -n "$win" "$cmd"; first=0
  else tmux new-window -t "=$SESSION:" -n "$win" "$cmd"; fi
  sleep 20   # stagger start-up, so lanes sharing a GPU see each other's memory in the check
done
echo "started $EXP: ${#LANES[@]} lane(s) in tmux session $SESSION"
echo "watch: tmux attach -t $SESSION   |   tail -f $(dirname "$CODE")/logs/$EXP.log"
