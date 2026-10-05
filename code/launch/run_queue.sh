#!/usr/bin/env bash
# Run one lane of an experiment on the lab server: a list of training runs, one after another, on one GPU.
#
#   bash code/launch/run_queue.sh code/experiments/E14/lane_gpu0_a.txt
#
# You normally don't call this directly: `bash code/launch/start.sh E14` opens one tmux window per
# lane file and runs this in each (so runs survive an ssh disconnect).
#
# An experiment is a folder code/experiments/<EXP>/ containing:
#   common.args          flags shared by every run (one or more per line), plus optional directives:
#                          @module models.llm_classifier   trainer to run (default: models.encoder)
#                          @hf-push                        upload each finished run to HF (tracking.py push)
#                          @need-gb 60                     GPU memory a run needs WITHOUT gradient
#                                                          checkpointing (llm_classifier only, see below)
#   lane_gpu<N>_<x>.txt  one lane each; the GPU is taken from the file name.
#                        Each non-comment line:   <run-name> <seed> [extra flags]
#                          <seed> is a number (-> runs/<name>/seed<k>/) or fold<k> for cross-fitting
#                          (-> --seed 0 --fold k, runs/<name>/fold<k>/)
#                        or a wait:   @after <run-name> <seed> [<seed> ...]
#                        which blocks the lane until those runs have finished.
#
# Lanes may list the SAME runs: each run is locked while it trains, so a lane that reaches a run another
# lane holds skips it. Finished runs (metrics.json) are skipped; interrupted ones resume from their last
# completed epoch (resume.pt). To move work, edit lane files and restart -- nothing is lost.
#
# GPU memory (llm_classifier). With `--no-grad-checkpoint` in common.args a run needs @need-gb free. If
# the GPU has less (a lab-mate's job), the lane waits up to 30 min, then starts with
# `--grad-checkpoint` instead: the computation is identical (same micro-batches, same order), only
# ~1.35x slower. An out-of-memory crash mid-run is retried the same way, resuming from its checkpoint.
# The encoder has its own memory check (FAIL FAST -> wait 10 min, retry, 2 h max).
#
# Encoder configurations are post-processed (analysis, decision rules, submission) by the lane that
# finishes their last seed.
#
# Never edit this file in place while lanes are running: bash reads a script as it executes, so a
# running lane can pick up a fragment of the new text. Write a new copy and `mv` it over this one.

set -u
LANE="$(readlink -f "$1")"
EXPDIR="$(dirname "$LANE")"
EXP="$(basename "$EXPDIR")"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$CODE")"
cd "$CODE"

GPU="$(basename "$LANE" | sed -nE 's/^lane_gpu([0-9]+)_.*\.txt$/\1/p')"
[ -n "$GPU" ] || { echo "lane file must be named lane_gpu<N>_<x>.txt: $LANE"; exit 2; }

PY="${CLARITY_PY:-/scratch/shlok/Temp/.venv/bin/python}"
# A lane file may pin an exact device with a line "# device: <id>" -- e.g. a MIG slice's UUID.
DEVICE="$(sed -nE 's/^#[[:space:]]*device:[[:space:]]*([^[:space:]]+).*/\1/p' "$LANE" | head -1)"
export CUDA_VISIBLE_DEVICES="${DEVICE:-$GPU}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS_FILE="$EXPDIR/common.args"
directive() { sed -nE "s/^@$1[[:space:]]*(.*)$/\1/p" "$ARGS_FILE" | head -1; }
MODULE="$(directive module)"; MODULE="${MODULE:-models.encoder}"
HF_PUSH=0; grep -qE '^@hf-push' "$ARGS_FILE" && HF_PUSH=1
NEED_GB="$(directive need-gb)"; NEED_GB="${NEED_GB:-60}"
COMMON="$(grep -vE '^\s*(#|$|@)' "$ARGS_FILE" | tr '\n' ' ')"
RUNS="${CLARITY_RUNS:-$ROOT/runs}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$RUNS" "$ROOT/docs/raw"
LOG="$LOGS/${EXP}.log"
TAG="$(basename "$LANE" .txt)"
log() { echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOG"; }

run_dir() { case "$2" in fold*) echo "$RUNS/$1/$2" ;; *) echo "$RUNS/$1/seed$2" ;; esac; }
seed_args() { case "$1" in fold*) echo "--seed 0 --fold ${1#fold}" ;; *) echo "--seed $1" ;; esac; }
lane_lines() { cat "$EXPDIR"/lane_gpu*_*.txt | grep -vE '^\s*(#|$|@)'; }
expected_seeds() { lane_lines | awk -v n="$1" '$1==n {print $2}' | sort -u | wc -l; }
finished_seeds() { ls "$RUNS/$1"/seed*/metrics.json "$RUNS/$1"/fold*/metrics.json 2>/dev/null | wc -l; }
is_subtask() { lane_lines | awk -v n="$1" '$1==n' | grep -qE -- "--task +(gate|other6|nr3|pair:)"; }
free_gb() {  # free memory of the lane's GPU in GB; empty if unknown (e.g. a MIG slice)
  [ -n "$DEVICE" ] && return 0
  nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1/1024}'
}

hf_push() {  # run dir -- background upload, one at a time (same lock as utils/tracking.py push_async)
  [ "$HF_PUSH" = 1 ] || return 0
  ( flock -w 14400 "$LOGS/.hf_upload.lock" timeout 3600 "$PY" "$CODE/utils/tracking.py" push "$1" \
      >> "$LOGS/hf_uploads.log" 2>&1 & )
  log "hf    upload queued: ${1#$ROOT/} (logs/hf_uploads.log)"
}

post_process() {  # name -- encoder only: analysis + decision rules + submission, once all seeds are in
  local name="$1"
  [ "$MODULE" = "models.encoder" ] || return 0
  is_subtask "$name" && return 0
  exec 9>"$RUNS/.post_${name}.lock"; flock 9
  if [ -f "$RUNS/$name/.post_done" ] || [ "$(finished_seeds "$name")" -lt "$(expected_seeds "$name")" ]; then
    flock -u 9; return 0
  fi
  log "post  $name: all $(expected_seeds "$name") seeds finished -- analysis, decision rules, submission"
  "$PY" -m evaluation.analyze --run-dir "$RUNS/$name" 2>/dev/null | grep -v -iE "warn|unauthenticated" \
      > "$ROOT/docs/raw/${name}_analysis.txt"
  "$PY" -m decision_rules.decide --run-dir "$RUNS/$name" --drop-annotator 2>/dev/null \
      | grep -v -iE "warn|unauthenticated|^written:" > "$ROOT/docs/raw/${name}_decision_rules.txt"
  "$PY" -m evaluation.make_submission --source stage1 --run "$name" --logit-adjust >> "$LOG" 2>&1
  touch "$RUNS/$name/.post_done"
  log "ok    $name post-processing -> docs/raw/${name}_{analysis,decision_rules}.txt"
  flock -u 9
}

memory_mode() {  # echo the checkpointing flag override for llm_classifier ("" = keep common.args)
  [ "$MODULE" = "models.llm_classifier" ] || return 0
  echo " $COMMON $* " | grep -q -- " --no-grad-checkpoint " || return 0
  local f w fell_back="$LOGS/.${EXP}_${TAG}.fell_back"
  for w in $(seq 0 "${QUEUE_WAIT_STEPS:-15}"); do   # 2-min steps
    f="$(free_gb)"; [ -z "$f" ] && return 0
    if [ "$f" -ge "$NEED_GB" ]; then rm -f "$fell_back"; return 0; fi
    [ -f "$fell_back" ] && break   # this lane already waited once: don't wait 30 min before every run
    [ "$w" = 0 ] && log "wait  GPU $GPU has ${f} GB free, a run without checkpointing needs ${NEED_GB} GB (up to 30 min)" >&2
    sleep 120
  done
  touch "$fell_back"
  log "note  GPU $GPU has only $(free_gb) GB free: this run uses --grad-checkpoint (same computation, slower)" >&2
  echo "--grad-checkpoint"
}

attempt() {  # name seed extra...
  local name="$1" seed="$2"; shift 2
  local out logf ckpt="" try rc
  out="$(run_dir "$name" "$seed")"; logf="$LOGS/${name}_$(basename "$out").log"
  [ -f "$out/metrics.json" ] && { log "skip  $name $seed (finished)"; return 0; }
  mkdir -p "$out"
  exec 8>"$out/.lane.lock"   # held (also by the trainer, which inherits it) until this run ends
  if ! flock -n 8; then log "skip  $name $seed (another lane is running it)"; exec 8>&-; return 1; fi
  [ -f "$out/metrics.json" ] && { log "skip  $name $seed (finished)"; exec 8>&-; return 0; }
  [ -f "$out/resume.pt" ] && log "resume $name $seed from its last checkpoint"
  for try in $(seq 1 12); do
    [ -z "$ckpt" ] && ckpt="$(memory_mode "$@")"
    log "start $name $seed on GPU $GPU ($MODULE${ckpt:+, $ckpt})"
    # shellcheck disable=SC2046,SC2086
    "$PY" -m "$MODULE" --name "$name" $(seed_args "$seed") --outroot "$RUNS" $COMMON "$@" $ckpt >>"$logf" 2>&1
    rc=$?
    if [ $rc = 0 ]; then
      log "ok    $name $seed  $(grep -oE 'devS2=[0-9.]+ devS1=[0-9.]+' "$logf" | tail -1)"
      hf_push "$out"; exec 8>&-; return 0
    fi
    if tail -3 "$logf" | grep -q "FAIL FAST"; then
      log "wait  $name $seed: GPU $GPU memory held by another job, retry $try/12 in 10 min"; sleep 600
    elif [ "$MODULE" = "models.llm_classifier" ] && tail -40 "$logf" | grep -qE "OutOfMemoryError|CUDA out of memory" \
         && [ "$ckpt" != "--grad-checkpoint" ]; then
      ckpt="--grad-checkpoint"
      log "oom   $name $seed: out of GPU memory -- resuming from the last epoch with --grad-checkpoint"
    elif [ "$try" -lt 2 ]; then
      log "retry $name $seed (exit $rc, see ${logf#$ROOT/}); resuming from its last finished epoch"
      tail -3 "$logf" | tee -a "$LOG"; sleep 60
    else
      log "FAIL  $name $seed (see ${logf#$ROOT/}):"; tail -4 "$logf" | tee -a "$LOG"; exec 8>&-; return 1
    fi
  done
  log "GAVE UP $name $seed after 12 attempts"; exec 8>&-; return 1
}

wait_for() {  # run-name seed... -- block until each has finished (metrics.json)
  local name="$1"; shift
  local s pending
  log "wait  for $name $* to finish before starting this lane's next run"
  while :; do
    pending=""; for s in "$@"; do [ -f "$(run_dir "$name" "$s")/metrics.json" ] || pending+=" $s"; done
    [ -z "$pending" ] && break
    sleep 300
  done
  log "done  waiting: $name $* finished"
}

log "lane start ($EXP, GPU $GPU, $MODULE): $(grep -cvE '^\s*(#|$|@)' "$LANE") run(s)"
while read -r name seed extra; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  # shellcheck disable=SC2086
  if [ "$name" = "@after" ]; then wait_for "$seed" $extra; continue; fi
  # shellcheck disable=SC2086
  attempt "$name" "$seed" $extra && post_process "$name"
done < <(grep -vE '^\s*(#|$)' "$LANE")

# Retry uploads that failed or never ran, for this experiment's runs only.
if [ "$HF_PUSH" = 1 ]; then
  for name in $(lane_lines | awk '{print $1}' | sort -u); do
    for d in "$RUNS/$name"/seed* "$RUNS/$name"/fold*; do
      [ -f "$d/metrics.json" ] && [ ! -f "$d/.hf_pushed" ] || continue
      flock -w 14400 "$LOGS/.hf_upload.lock" timeout 3600 "$PY" "$CODE/utils/tracking.py" push "$d" >> "$LOGS/hf_uploads.log" 2>&1
    done
  done
  log "hf    $(ls "$RUNS"/*/{seed,fold}*/.hf_pushed 2>/dev/null | wc -l) run folders marked uploaded"
fi
log "lane finished"
