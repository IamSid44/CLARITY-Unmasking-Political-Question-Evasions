#!/usr/bin/env bash
# The full pipeline, as one resumable job. Launch it inside tmux:
#
#     tmux new-session -d -s clarity "bash clarity/run_pipeline.sh"
#     tmux attach -t clarity          # watch;  Ctrl-b d  to detach again
#
# Stages, in order, each producing something usable on its own:
#   A  Stage-1 baseline   DeBERTa-v3-large, flat 9-way, 5 seeds   ~2.5 h
#      -> submissions/L0_large_base/          (Codabench-ready, both subtasks)
#      -> analysis + decision rules on the baseline (CPU)
#   B  Cross-fitted folds  same model, 5-fold on train, OOF probs  ~1.8 h
#   C  Stage-2 re-ranker  seed 0                                   ~1 h
#      -> submissions/R1_rerank_s0_prior/     (first re-ranked submission)
#   D  Stage-2 re-ranker  seeds 1-4                                ~4 h
#      -> submissions/R1_rerank/ and R1_rerank_prior/  (5-seed)
#
# Checkpointed at two levels, so a server restart costs at most one epoch:
#   * a run with metrics.json is complete and is skipped;
#   * a run with resume.pt was interrupted and continues from its last completed
#     epoch -- model, optimizer, scheduler, RNG and data order restored (ckpt.py);
#   * HF uploads that failed or were interrupted are retried by `push-pending`
#     between stages (a successful upload leaves a .hf_pushed marker).
# After a reboot:   bash clarity/start.sh   (same command as the first launch)
#
# CLAUDE.md invariant 5: GPU 0 only. If a neighbour has taken the memory, the
# run fails fast (never OOMs someone else's job); this script then waits 10 min
# and retries, up to 2 h, instead of silently skipping it.

set -u
cd "$(dirname "$0")"
PY=/scratch/shlok/Temp/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs submissions
LOG=logs/pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SEEDS="0 1 2 3 4"
# 8 epochs. A 5-epoch schedule was tried (2026-09-19) and was UNDERTRAINED: the
# cosine LR decays over the whole run, so the 8-epoch run's epoch-3 peak happened
# at a still-high LR that a 5-epoch run never gets. 5-epoch seeds ended at train
# loss 1.29-1.48 with best val 0.27-0.37 and were still improving at the last
# epoch, vs 0.58 / 0.43 for 8 epochs. Kept in runs/_exploratory/. Selection is by
# internal val, so the extra epochs cannot hurt the selected checkpoint.
S1="--model microsoft/deberta-v3-large --epochs 8 --batch-size 16 --max-vram-gb 20 --lr 1e-5 --head-lr 1e-4 --llrd 0.95"
# --epochs is NOT passed here: rerank.py owns that default (8). Passing it
# explicitly is how the 3-epoch undertrained re-ranker survived a "fix" that only
# changed the default -- the flag silently overrode it.
RR="--model microsoft/deberta-v3-large --items-per-batch 4 --grad-accum 4 --max-vram-gb 20"

attempt() {  # name  logfile  command...
  local name="$1" logf="$2"; shift 2
  for try in $(seq 1 12); do
    "$@" >>"$logf" 2>&1 && { log "ok    $name"; return 0; }
    if tail -3 "$logf" | grep -q "FAIL FAST"; then
      log "wait  $name: GPU memory taken by another job, retry $try/12 in 10 min"; sleep 600
    else
      log "FAIL  $name (see $logf):"; tail -4 "$logf" | tee -a "$LOG"; return 1
    fi
  done
  log "GAVE UP $name after 2 h of waiting for GPU memory"; return 1
}

done_() { [ -f "$1/metrics.json" ]; }

log "pipeline start"
push_pending() { $PY tracking.py push-pending runs >>"$LOG" 2>&1; }
push_pending

# --- A. Stage-1 baseline ------------------------------------------------------
for s in $SEEDS; do
  done_ runs/L0_large_base/seed$s && { log "skip  L0_large_base seed$s"; continue; }
  attempt "L0_large_base seed$s" logs/L0_large_base_seed$s.log \
    $PY encoder.py --name L0_large_base --seed $s $S1 --save-model --push-to-hub
done
$PY make_submission.py --source stage1 --run L0_large_base >>"$LOG" 2>&1 && log "ok    submission L0_large_base"
$PY analyze.py --run-dir runs/L0_large_base > reports/E0_analysis.txt 2>&1
$PY decide.py --run-dir runs/L0_large_base --drop-annotator > reports/E0_decision_rules.txt 2>&1
log "ok    E0 analysis -> reports/E0_analysis.txt, reports/E0_decision_rules.txt"
push_pending

# --- B. Cross-fitted folds (out-of-fold candidates for the re-ranker) --------
for f in 0 1 2 3 4; do
  done_ runs/F0_oof/fold$f && { log "skip  F0_oof fold$f"; continue; }
  attempt "F0_oof fold$f" logs/F0_oof_fold$f.log \
    $PY encoder.py --name F0_oof --fold $f $S1 --select last
done

# --- C/D. Stage-2 re-ranker ---------------------------------------------------
for s in $SEEDS; do
  done_ runs/R1_rerank/seed$s && { log "skip  R1_rerank seed$s"; continue; }
  attempt "R1_rerank seed$s" logs/R1_rerank_seed$s.log \
    $PY rerank.py --seed $s $RR --push-to-hub
  if [ "$s" = "0" ]; then
    $PY make_submission.py --source rerank --run R1_rerank --with-prior --tag R1_rerank_s0_prior >>"$LOG" 2>&1 \
      && log "ok    submission R1_rerank_s0_prior (1 seed)"
  fi
done
push_pending
$PY make_submission.py --source rerank --run R1_rerank >>"$LOG" 2>&1 && log "ok    submission R1_rerank"
$PY make_submission.py --source rerank --run R1_rerank --with-prior >>"$LOG" 2>&1 && log "ok    submission R1_rerank_prior"
log "pipeline finished"
