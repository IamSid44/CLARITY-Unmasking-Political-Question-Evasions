#!/usr/bin/env bash
# Start -- or, after a reboot, resume -- an experiment in tmux.
#
#   bash clarity/start.sh E8          # every lane in clarity/experiments/E8/, one tmux window each
#   bash clarity/start.sh pipeline    # the original E0 + E6 pipeline (run_pipeline.sh)
#
#   tmux attach -t clarity-E8         # watch;  Ctrl-b n / Ctrl-b p to switch lanes;  Ctrl-b d to detach
#   tail -f clarity/logs/E8.log       # one line per event, all lanes
#
# Safe to run twice: it refuses to start a second copy of something already running.
# Everything inside is resumable, so after a reboot this same command continues
# where the experiment stopped.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
EXP="${1:-}"
[ -n "$EXP" ] || { sed -n '2,12p' "$0"; exit 2; }

if [ "$EXP" = "pipeline" ]; then
  if pgrep -f "bash .*run_pipeline\.sh" >/dev/null; then
    echo "pipeline already running -- attach with: tmux attach -t clarity"; exit 0
  fi
  tmux kill-session -t clarity 2>/dev/null
  tmux new-session -d -s clarity "bash $DIR/run_pipeline.sh; echo PIPELINE_EXITED; exec bash"
  echo "started. watch: tmux attach -t clarity"; exit 0
fi

EXPDIR="$DIR/experiments/$EXP"
LANES=( "$EXPDIR"/lane_gpu*_*.txt )
[ -e "${LANES[0]}" ] || { echo "no lane files in $EXPDIR"; exit 2; }
if pgrep -f "^bash .*/run_queue\.sh .*/experiments/$EXP/" >/dev/null; then   # anchored: a leftover `tmux new-session` client carries the same path and must not count
  echo "$EXP already running -- attach with: tmux attach -t clarity-$EXP"; exit 0
fi
SESSION="clarity-$EXP"
tmux kill-session -t "$SESSION" 2>/dev/null
first=1
for lane in "${LANES[@]}"; do
  win="$(basename "$lane" .txt)"
  # tmux does not inherit these from the calling shell; pass them through if set
  envp=""; for v in CLARITY_RUNS WANDB_MODE; do [ -n "${!v:-}" ] && envp+="$v=${!v} "; done
  cmd="env ${envp}bash $DIR/run_queue.sh $lane; echo LANE_EXITED; exec bash"
  if [ $first = 1 ]; then tmux new-session -d -s "$SESSION" -n "$win" "$cmd"; first=0
  else tmux new-window -t "$SESSION" -n "$win" "$cmd"; fi
  sleep 20   # stagger start-up, so lanes sharing a GPU see each other's memory in the check
done
echo "started $EXP: ${#LANES[@]} lane(s) in tmux session $SESSION"
echo "watch: tmux attach -t $SESSION   |   tail -f $DIR/logs/$EXP.log"
