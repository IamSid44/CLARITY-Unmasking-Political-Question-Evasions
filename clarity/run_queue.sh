#!/usr/bin/env bash
# Run one lane of an experiment: a list of training runs, one after another, on one GPU.
#
#   bash clarity/run_queue.sh clarity/experiments/E8/lane_gpu1_a.txt
#
# You normally don't call this directly: `bash clarity/start.sh E8` opens one tmux
# window per lane file and runs this in each.
#
# An experiment is a folder clarity/experiments/<EXP>/ containing:
#   common.args          flags shared by every run in the experiment (one per line is fine)
#   lane_gpu<N>_<x>.txt  one lane each; the GPU is taken from the file name.
#                        Each non-comment line:   <run-name> <seed> [extra flags]
#                        or a wait:   @after <run-name> <seed> [<seed> ...]
#                        which blocks the lane until those runs have finished --
#                        for queueing behind another experiment on the same GPU,
#                        since two runs sharing a GPU finish no sooner than in turn.
#
# Several lanes may share a GPU (lane_gpu1_a, lane_gpu1_b). To move work off a GPU,
# move lines between lane files and restart: finished runs are skipped and
# interrupted ones resume from their last epoch, so nothing is lost.
#
# Lanes may also list the SAME runs (e.g. one lane seeds 0..3, another 3..0): each
# run is locked while it trains, so a lane that reaches a run another lane holds
# skips it and moves on. The lanes then share the work between them, and whichever
# GPU is faster ends up doing more of it.
#
# Never edit this file in place while lanes are running: bash reads a script as it
# executes, so a running lane can pick up a fragment of the new text. Write a new
# copy and `mv` it over this one; running lanes keep reading the old file.
#
# Checkpointing, at three levels:
#   * run finished (metrics.json)        -> skipped
#   * run interrupted (resume.pt)        -> continues from its last completed epoch
#   * upload failed or interrupted       -> retried in the background (tracking.py)
# If another job holds the GPU's memory, a run refuses to start rather than risk an
# out-of-memory crash on a shared card; this script waits 10 min and retries (2 h max).
#
# When every seed of a configuration has finished (across ALL lanes), the lane that
# finishes last writes its analysis, decision rules and Codabench-format submission.

set -u
LANE="$(readlink -f "$1")"
EXPDIR="$(dirname "$LANE")"
EXP="$(basename "$EXPDIR")"
CLARITY="$(cd "$(dirname "$0")" && pwd)"
cd "$CLARITY"

GPU="$(basename "$LANE" | sed -nE 's/^lane_gpu([0-9]+)_.*\.txt$/\1/p')"
[ -n "$GPU" ] || { echo "lane file must be named lane_gpu<N>_<x>.txt: $LANE"; exit 2; }

PY=/scratch/shlok/Temp/.venv/bin/python
# A lane file may pin an exact device with a line "# device: <id>" -- e.g. a MIG slice's
# UUID (`nvidia-smi -L`). GPU 1 is split into two MIG slices, and CUDA_VISIBLE_DEVICES=1
# reaches only the first; the second needs its UUID.
DEVICE="$(sed -nE 's/^#[[:space:]]*device:[[:space:]]*([^[:space:]]+).*/\1/p' "$LANE" | head -1)"
export CUDA_VISIBLE_DEVICES="${DEVICE:-$GPU}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="$(grep -vE '^\s*(#|$)' "$EXPDIR/common.args" | tr '\n' ' ')"
RUNS="${CLARITY_RUNS:-$CLARITY/runs}"
mkdir -p logs "$RUNS" reports/raw
LOG="logs/${EXP}.log"
TAG="$(basename "$LANE" .txt)"
log() { echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOG"; }

# How many seeds a configuration has, across every lane of the experiment.
expected_seeds() { cat "$EXPDIR"/lane_gpu*_*.txt | grep -vE '^\s*(#|$)' | awk -v n="$1" '$1==n {print $2}' | sort -u | wc -l; }
finished_seeds() { ls "$RUNS/$1"/seed*/metrics.json 2>/dev/null | wc -l; }

# E12 sub-task models (gate, specialists, boundary experts) output probabilities over
# their own classes, not the 9 labels, so the 9-way post-processing does not apply to
# them; hier_combine.py scores them inside the combined system.
is_subtask() { cat "$EXPDIR"/lane_gpu*_*.txt | awk -v n="$1" '$1==n' | grep -qE -- "--task +(gate|other6|nr3|pair:)"; }

post_process() {  # name -- analysis + decision rules + submission, once, when all seeds are in
  local name="$1"
  if is_subtask "$name"; then return; fi
  exec 9>"$RUNS/.post_${name}.lock"; flock 9
  if [ -f "$RUNS/$name/.post_done" ]; then flock -u 9; return; fi
  if [ "$(finished_seeds "$name")" -lt "$(expected_seeds "$name")" ]; then flock -u 9; return; fi
  log "post  $name: all $(expected_seeds "$name") seeds finished -- analysis, decision rules, submission"
  $PY analyze.py --run-dir "$RUNS/$name" 2>/dev/null | grep -v -iE "warn|unauthenticated" > "reports/raw/${name}_analysis.txt"
  $PY decide.py --run-dir "$RUNS/$name" --drop-annotator 2>/dev/null \
      | grep -v -iE "warn|unauthenticated|^written:" > "reports/raw/${name}_decision_rules.txt"
  $PY make_submission.py --source stage1 --run "$name" --logit-adjust >> "$LOG" 2>&1
  touch "$RUNS/$name/.post_done"
  log "ok    $name post-processing -> reports/raw/${name}_{analysis,decision_rules}.txt, submissions/${name}_logitadj/"
  flock -u 9
}

attempt() {  # name seed extra...
  local name="$1" seed="$2"; shift 2
  local out="$RUNS/$name/seed$seed" logf="logs/${name}_seed${seed}.log"
  [ -f "$out/metrics.json" ] && { log "skip  $name seed$seed (finished)"; return 0; }
  mkdir -p "$out"
  exec 8>"$out/.lane.lock"   # held (also by the trainer, which inherits it) until this run ends
  if ! flock -n 8; then log "skip  $name seed$seed (another lane is running it)"; exec 8>&-; return 1; fi
  # a run another lane finished while this one waited for the lock
  [ -f "$out/metrics.json" ] && { log "skip  $name seed$seed (finished)"; exec 8>&-; return 0; }
  [ -f "$out/resume.pt" ] && log "resume $name seed$seed from its last checkpoint"
  for try in $(seq 1 12); do
    log "start $name seed$seed on GPU $GPU"
    # shellcheck disable=SC2086
    if $PY encoder.py --name "$name" --seed "$seed" --outroot "$RUNS" $COMMON "$@" >>"$logf" 2>&1; then
      log "ok    $name seed$seed  $(grep -oE 'devS2=[0-9.]+ devS1=[0-9.]+' "$logf" | tail -1)"
      exec 8>&-; return 0
    fi
    if tail -3 "$logf" | grep -q "FAIL FAST"; then
      log "wait  $name seed$seed: GPU $GPU memory held by another job, retry $try/12 in 10 min"; sleep 600
    else
      log "FAIL  $name seed$seed (see $logf):"; tail -4 "$logf" | tee -a "$LOG"; exec 8>&-; return 1
    fi
  done
  log "GAVE UP $name seed$seed after 2 h waiting for GPU memory"; exec 8>&-; return 1
}

wait_for() {  # run-name seed... -- block until each seed has finished (metrics.json)
  local name="$1"; shift
  local s pending
  log "wait  for $name seed(s) $* to finish before starting this lane's next run"
  while :; do
    pending=""; for s in "$@"; do [ -f "$RUNS/$name/seed$s/metrics.json" ] || pending+=" $s"; done
    [ -z "$pending" ] && break
    sleep 300
  done
  log "done  waiting: $name seed(s) $* finished"
}

log "lane start ($EXP, GPU $GPU): $(grep -cvE '^\s*(#|$|@)' "$LANE") run(s)"
while read -r name seed extra; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  # shellcheck disable=SC2086
  if [ "$name" = "@after" ]; then wait_for "$seed" $extra; continue; fi
  # shellcheck disable=SC2086
  attempt "$name" "$seed" $extra && post_process "$name"
done < <(grep -vE '^\s*(#|$)' "$LANE")
# Retry any failed uploads -- only if this experiment uploads at all, and only its own runs.
if grep -q -- "--push-to-hub" "$EXPDIR/common.args"; then
  NAMES="$(cat "$EXPDIR"/lane_gpu*_*.txt | grep -vE '^\s*(#|$)' | awk '{print $1}' | sort -u | tr '\n' ' ')"
  # shellcheck disable=SC2086
  $PY tracking.py push-pending "$RUNS" $NAMES >> "$LOG" 2>&1
fi
log "lane finished"
