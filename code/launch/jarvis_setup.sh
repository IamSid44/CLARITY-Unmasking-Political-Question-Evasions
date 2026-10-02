#!/usr/bin/env bash
set -uo pipefail

BASE=${BASE:-/home/clarity_llm}
CODE=$BASE/code
VENV=$BASE/venv
PY=$VENV/bin/python
LOGS=$BASE/logs
RUNS=$BASE/runs
KEYS=$BASE/keys.env
NAME=${NAME:-Q8_fullq_lora}
SEEDS=${SEEDS:-"0 1 2"}
MODEL=${MODEL:-Qwen/Qwen3-8B-Base}
SMOKE_MODEL=Qwen/Qwen3-0.6B-Base
EXTRA_ARGS=${EXTRA_ARGS:-}
MAX_TOTAL_HOURS=${MAX_TOTAL_HOURS:-5}
AUTO_PAUSE=${AUTO_PAUSE:-1}
PAUSE_DELAY=${PAUSE_DELAY:-60}
NO_PROMPT=${NO_PROMPT:-0}
SESSION=clarity-llm
SELF=$(readlink -f "$0")

export HF_HOME=$BASE/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() { mkdir -p "$LOGS"; echo "$(date '+%F %T') $*" | tee -a "$LOGS/pipeline.log"; }
die()   { stage "FAILED: $*"; maybe_pause "failure"; exit 1; }
load_keys() { [ -f "$KEYS" ] && set -a && . "$KEYS" && set +a; return 0; }

bundle() {
  cd "$(dirname "$SELF")/../.."
  [ -f code/models/llm_classifier.py ] || { echo "code/models/llm_classifier.py not found"; exit 2; }
  [ -f data/splits/train_internal_val_index.json ] || { echo "missing data/splits"; exit 2; }
  [ -f data/cache/train.parquet ] || (cd code && python -c "from qevasion.loader import load_qevasion; load_qevasion()")
  tar czf jarvis_bundle.tgz --exclude='__pycache__' \
    code/launch/jarvis_setup.sh \
    code/models/__init__.py code/models/llm_classifier.py code/utils/__init__.py code/utils/tracking.py code/qevasion \
    data/cache/train.parquet data/cache/dev.parquet data/clarity_task_evaluation_dataset.csv \
    data/splits/train_internal_val_index.json
  ls -lh jarvis_bundle.tgz && tar tzf jarvis_bundle.tgz
}

ask_keys() {
  [ -f "$KEYS" ] && { echo "keys already in $KEYS (delete it to re-enter)"; return; }
  [ "$NO_PROMPT" = 1 ] && { echo "NO_PROMPT=1: no keys file, so no auto-pause from the instance"; return; }
  echo "== Optional keys (input hidden; press Enter to skip any) =="
  echo "   JarvisLabs API key + this instance's machine id -> the instance pauses itself when done"
  read -rsp "JL_API_KEY (jarvislabs.ai/settings/api-keys): " JL_API_KEY; echo
  read -rp  "JL_MACHINE_ID (shown on the instance page / 'jl list'): " JL_MACHINE_ID
  echo "   Hugging Face write token + repo -> results are also uploaded there before pausing"
  read -rsp "HF_TOKEN (write access): " HF_TOKEN; echo
  read -rp  "HF_REPO_ID (e.g. someone/clarity-semeval26): " HF_REPO_ID
  mkdir -p "$BASE"; umask 077
  printf 'JL_API_KEY=%s\nJL_MACHINE_ID=%s\nHF_TOKEN=%s\nHF_REPO_ID=%s\n' \
    "$JL_API_KEY" "$JL_MACHINE_ID" "$HF_TOKEN" "$HF_REPO_ID" > "$KEYS"
  chmod 600 "$KEYS"; echo "wrote $KEYS (mode 600)"
}

check_pause_keys() {
  load_keys
  if [ "$AUTO_PAUSE" != 1 ] || [ -z "${JL_API_KEY:-}" ] || [ -z "${JL_MACHINE_ID:-}" ]; then
    stage "auto-pause OFF (AUTO_PAUSE=$AUTO_PAUSE, key/id set: ${JL_API_KEY:+yes}/${JL_MACHINE_ID:+yes}) -- pause by hand when done"
    return
  fi
  if "$PY" - <<'EOF'
import os
from jarvislabs import Client
print(Client().instances.get(int(os.environ["JL_MACHINE_ID"])))
EOF
  then stage "auto-pause ON: key valid, machine $JL_MACHINE_ID found"
  else stage "auto-pause check FAILED (see above) -- the instance will NOT pause itself; pause by hand"; AUTO_PAUSE=0
  fi
}

maybe_pause() {
  load_keys
  [ "$AUTO_PAUSE" = 1 ] && [ -n "${JL_API_KEY:-}" ] && [ -n "${JL_MACHINE_ID:-}" ] && [ -x "$PY" ] || return 0
  stage "pausing instance $JL_MACHINE_ID ($1) in $PAUSE_DELAY s -- Ctrl-C in tmux to keep it running"
  sleep "$PAUSE_DELAY"
  "$PY" -c "import os; from jarvislabs import Client; Client().instances.pause(int(os.environ['JL_MACHINE_ID']))" \
    || "$VENV/bin/jl" pause "$JL_MACHINE_ID" --yes || stage "pause call failed -- pause by hand"
}

setup() {
  stage "setup: start"
  [ -f "$CODE/code/models/llm_classifier.py" ] || die "no code in $CODE -- extract jarvis_bundle.tgz there first"
  mkdir -p "$LOGS" "$RUNS" "$HF_HOME"
  {
    SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO=sudo
    if ! ping -c 2 -W 2 -M do -s 1472 1.1.1.1 >/dev/null 2>&1 && ping -c 2 -W 2 -M do -s 1400 1.1.1.1 >/dev/null 2>&1; then
      dev=$(ip route show default | awk '{print $5; exit}')
      $SUDO ip link set dev "$dev" mtu 1450 && $SUDO sysctl -q -w net.ipv4.tcp_mtu_probing=1 && echo "MTU of $dev lowered to 1450"
    fi
    curl -6 -s -o /dev/null --max-time 8 https://huggingface.co || {
      $SUDO sysctl -q -w net.ipv6.conf.all.disable_ipv6=1 net.ipv6.conf.default.disable_ipv6=1 && echo "IPv6 disabled"; }
    command -v tmux >/dev/null && command -v curl >/dev/null || {
      SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO=sudo
      $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux curl >/dev/null; }
    command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    [ -x "$PY" ] || uv venv "$VENV" --python 3.12

    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if   [ "$DRV" -ge 570 ]; then CU=cu128; TORCH="torch==2.11.0"
    elif [ "$DRV" -ge 560 ]; then CU=cu126; TORCH="torch==2.11.0"
    else CU=cu124; TORCH="torch"; fi
    echo "driver $DRV -> $TORCH from $CU"
    "$PY" -c "import torch" 2>/dev/null || uv pip install --python "$PY" "$TORCH" --index-url "https://download.pytorch.org/whl/$CU"
    uv pip install --python "$PY" "transformers==5.17.0" "peft==0.20.0" "accelerate==1.15.0" \
      "numpy==2.5.2" "pandas==3.0.5" "pyarrow==25.0.1" "huggingface_hub>=1.28" jarvislabs wandb python-dotenv
  } > "$LOGS/setup.log" 2>&1 || die "install failed (logs/setup.log)"

  "$PY" - >> "$LOGS/setup.log" 2>&1 <<'EOF' || die "GPU check failed (logs/setup.log)"
import torch
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); (x @ x).sum().item()
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} {p.total_memory / 1024**3:.0f} GiB | bf16 {torch.cuda.is_bf16_supported()}")
EOF
  stage "setup: $(tail -1 "$LOGS/setup.log")"

  stage "setup: downloading $MODEL and $SMOKE_MODEL to $HF_HOME"
  "$PY" - >> "$LOGS/setup.log" 2>&1 <<EOF || die "model download failed (logs/setup.log)"
from huggingface_hub import snapshot_download
for m in ("$SMOKE_MODEL", "$MODEL"):
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
EOF
  stage "setup: done ($(du -sh "$HF_HOME" | cut -f1) of models; disk free $(df -h /home | awk 'NR==2{print $4}'))"
}

llm() { cd "$CODE/code" && "$PY" -m models.llm_classifier "$@"; }

smoke() {
  stage "smoke: start ($SMOKE_MODEL, 64 rows, 2 epochs, stop after epoch 0, then resume)"
  rm -rf "$BASE/smoke"
  local A=(--name SMOKE --seed 0 --model "$SMOKE_MODEL" --limit-train 64 --limit-eval 32 --epochs 2
           --log-every 4 --outroot "$BASE/smoke")
  { llm "${A[@]}" --stop-after-epoch 0 && llm "${A[@]}"; } > "$LOGS/smoke.log" 2>&1 || die "smoke failed (logs/smoke.log)"
  grep -q "\[resume\] continuing after epoch 0" "$LOGS/smoke.log" || die "smoke: resume path not exercised"
  [ -f "$BASE/smoke/SMOKE/seed0/metrics.json" ] || die "smoke: no metrics.json"
  stage "smoke: passed ($(grep '\[check\]' "$LOGS/smoke.log" | head -1 | sed 's/.*= //'))"
}

pilot() {
  local plog="$LOGS/pilot_$NAME.log"
  stage "pilot: 20 optimizer steps of $MODEL ($NAME, EXTRA_ARGS='$EXTRA_ARGS')"
  llm --name PILOT --seed 0 --model "$MODEL" --pilot-steps 20 --outroot "$BASE/pilot" $EXTRA_ARGS \
    > "$plog" 2>&1 || die "pilot failed (logs/pilot_$NAME.log) -- often out of memory: try EXTRA_ARGS='--batch-size 2 --grad-accum 8'"
  grep '^\[pilot\] longest' "$plog" | sed 's/^\[pilot\] /pilot: /' | while read -r l; do stage "$l"; done
  local line min_ep epochs n hours
  line=$(grep '^\[pilot\] [0-9]' "$plog") || die "pilot printed no timing"
  stage "pilot: ${line#\[pilot\] }"
  min_ep=$(echo "$line" | sed -E 's/.*-> ([0-9.]+) min per epoch.*/\1/')
  epochs=$(echo " $EXTRA_ARGS " | sed -nE 's/.* --epochs ([0-9]+) .*/\1/p'); epochs=${epochs:-3}
  n=$(echo $SEEDS | wc -w)
  hours=$(awk -v m="$min_ep" -v e="$epochs" -v n="$n" 'BEGIN{printf "%.1f", n*(e*(m+1)+3)/60}')
  stage "pilot: projected $hours h for $n seeds x $epochs epochs (limit MAX_TOTAL_HOURS=$MAX_TOTAL_HOURS)"
  awk -v h="$hours" -v lim="$MAX_TOTAL_HOURS" 'BEGIN{exit !(h>lim)}' && die "projected time over the limit"
  return 0
}

train() {
  for s in $SEEDS; do
    if [ -f "$RUNS/$NAME/seed$s/metrics.json" ]; then stage "train: seed $s already finished"; continue; fi
    for attempt in 1 2; do
      stage "train: seed $s attempt $attempt"
      if llm --name "$NAME" --seed "$s" --model "$MODEL" --outroot "$RUNS" --track $EXTRA_ARGS >> "$LOGS/${NAME}_seed$s.log" 2>&1; then
        stage "train: seed $s done: $(grep '^\[done\]' "$LOGS/${NAME}_seed$s.log" | tail -1)"
        hf_push_bg "$RUNS/$NAME/seed$s"; break
      fi
      [ "$attempt" = 2 ] && die "seed $s failed twice (logs/${NAME}_seed$s.log)"
      stage "train: seed $s failed; retrying from its last finished epoch"
    done
  done
}

HF_LOCK=$LOGS/.hf.lock
hf_push_bg() {
  [ -f "$CODE/code/.env" ] || { stage "hf: no code/.env on the instance -- upload skipped"; return 0; }
  ( flock -w 14400 "$HF_LOCK" timeout 3600 "$PY" "$CODE/code/utils/tracking.py" push "$1" >> "$LOGS/hf_uploads.log" 2>&1 & )
}
hf_flush() {
  [ -f "$CODE/code/.env" ] || return 0
  local d n_up n_all
  for d in "$RUNS/$NAME"/seed*; do
    [ -f "$d/metrics.json" ] || continue
    flock -w 7200 "$HF_LOCK" bash -c "[ -f '$d/.hf_pushed' ] || timeout 3600 '$PY' '$CODE/code/utils/tracking.py' push '$d'" \
      >> "$LOGS/hf_uploads.log" 2>&1
  done
  n_up=$(ls "$RUNS/$NAME"/seed*/.hf_pushed 2>/dev/null | wc -l); n_all=$(ls "$RUNS/$NAME"/seed*/metrics.json 2>/dev/null | wc -l)
  stage "hf: $n_up of $n_all seed folders of $NAME on the Hub (logs/hf_uploads.log)"
}

summary() {
  "$PY" - "$RUNS/$NAME" <<'EOF' | tee "$LOGS/summary.txt"
import json, sys
from pathlib import Path
import numpy as np
REF = {0: (0.344, 0.591), 1: (0.370, 0.627), 2: (0.428, 0.640)}
run = Path(sys.argv[1]); rows = []
print(f"{run.name}: single models on dev (multi-reference S2, S1), epoch chosen on the train slice")
print(f"{'seed':>4} {'epoch':>5} {'slice F1':>8} {'dev S2':>7} {'dev S1':>7}   DeBERTa S2/S1   per-epoch dev S2")
for m in sorted(run.glob("seed*/metrics.json")):
    r = json.loads(m.read_text()); s = r["config"]["seed"]
    for k in ("dev_subtask2_macro_f1", "dev_subtask1_macro_f1", "val_macro_f1"):
        r[k] = float("nan") if r.get(k) is None else r[k]
    hist = " ".join(f"{h.get('dev_subtask2_macro_f1', float('nan')):.3f}" for h in r["history"])
    ref = REF.get(s, (float("nan"),) * 2)
    print(f"{s:>4} {r['selected_epoch']:>5} {r['val_macro_f1']:>8.3f} {r['dev_subtask2_macro_f1']:>7.3f} "
          f"{r['dev_subtask1_macro_f1']:>7.3f}   {ref[0]:.3f}/{ref[1]:.3f}     {hist}")
    rows.append((s, r["dev_subtask2_macro_f1"], r["dev_subtask1_macro_f1"], *ref, r["wall_minutes"], r.get("peak_vram_gb") or 0))
if rows:
    a = np.array(rows, dtype=float)
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"\nmean S2 {a[:,1].mean():.3f} ± {sd(a[:,1]):.3f}   S1 {a[:,2].mean():.3f} ± {sd(a[:,2]):.3f}")
    d2, d1 = a[:,1] - a[:,3], a[:,2] - a[:,4]
    print(f"paired vs DeBERTa (same seeds): S2 {d2.mean():+.3f} ({(d2 > 0).sum()}/{len(d2)} up), "
          f"S1 {d1.mean():+.3f} ({(d1 > 0).sum()}/{len(d1)} up)")
    print("screening bar (experiment log, E12): extend to more seeds if >= +0.015 with >= 2 of 3 seeds up")
    print(f"wall {a[:,5].mean():.0f} min per seed; peak VRAM {a[:,6].max():.1f} GiB")
EOF
}

package() {
  cd "$BASE" || exit 1
  tar czf "results_$NAME.tgz" --exclude='resume.pt' --exclude='adapter_best' --exclude='ep*_adapter' "runs/$NAME" logs
  tar czf "adapters_$NAME.tgz" runs/"$NAME"/seed*/adapter_best 2>/dev/null || true
  stage "package: $(ls -lh results_"$NAME".tgz | awk '{print $5}') results, $(ls -lh adapters_"$NAME".tgz 2>/dev/null | awk '{print $5}') adapters in $BASE"
  load_keys
  if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO_ID:-}" ]; then
    "$PY" - "$BASE" "$NAME" <<'EOF' && stage "package: uploaded to HF $HF_REPO_ID under jarvis/" || stage "package: HF upload failed (files are still on the instance)"
import os, sys
from huggingface_hub import HfApi
base, name = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
for f in (f"results_{name}.tgz", f"adapters_{name}.tgz"):
    if os.path.exists(f"{base}/{f}"):
        api.upload_file(path_or_fileobj=f"{base}/{f}", path_in_repo=f"jarvis/{f}", repo_id=os.environ["HF_REPO_ID"])
EOF
  fi
}

status() {
  echo "== stages"; tail -n 15 "$LOGS/pipeline.log" 2>/dev/null
  echo "== latest training lines"; tail -n 4 "$LOGS"/"${NAME}"_seed*.log 2>/dev/null
  echo "== finished seeds"; ls "$RUNS/$NAME"/seed*/metrics.json 2>/dev/null || echo "none yet"
  echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session $SESSION: running (tmux attach -t $SESSION)" \
    || echo "tmux session $SESSION: not running"
}

pipeline() {
  stage "pipeline: start (NAME=$NAME SEEDS='$SEEDS' MODEL=$MODEL EXTRA_ARGS='$EXTRA_ARGS')"
  [ -x "$PY" ] && [ -f "$LOGS/.setup_done" ] || { setup && touch "$LOGS/.setup_done"; }
  check_pause_keys
  [ -f "$LOGS/.smoke_done" ] || { smoke && touch "$LOGS/.smoke_done"; }
  [ -f "$LOGS/.pilot_done_$NAME" ] || { pilot && touch "$LOGS/.pilot_done_$NAME"; }
  train
  hf_flush
  summary
  package
  stage "pipeline: ALL DONE"
  maybe_pause "finished"
}

case "${1:-}" in
  bundle)  bundle ;;
  all)
    ask_keys
    command -v tmux >/dev/null || { SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO=sudo; $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux >/dev/null; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "already running: tmux attach -t $SESSION"; exit 0; }
    tmux new-session -d -s "$SESSION" "NAME='$NAME' SEEDS='$SEEDS' MODEL='$MODEL' EXTRA_ARGS='$EXTRA_ARGS' MAX_TOTAL_HOURS=$MAX_TOTAL_HOURS AUTO_PAUSE=$AUTO_PAUSE PAUSE_DELAY=$PAUSE_DELAY bash '$SELF' _pipeline; exec bash"
    echo "started in tmux session '$SESSION'. Watch:  tmux attach -t $SESSION   (Ctrl-b d to leave)"
    echo "                                  or:  tail -f $LOGS/pipeline.log"
    ;;
  _pipeline) pipeline ;;
  setup)   setup ;;
  smoke)   smoke ;;
  pilot)   pilot ;;
  train)   train ;;
  summary) summary ;;
  package) package ;;
  status)  status ;;
  *) sed -n '2,32p' "$SELF"; exit 2 ;;
esac
