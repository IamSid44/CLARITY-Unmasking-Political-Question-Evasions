#!/usr/bin/env bash
# E26, NF4 half, on the shared lab server (phoenix; 3 x RTX A4000). Everything lives in ~/e26_clarity; only GPUs that
# carry NO other process are used (GPU 1 runs another user's job and is never touched). Phases:
#   0 setup: venv, packages, adapters (public Hub repo), base model, frames
#   1 NF4: seeds 0-4 on GPU 0 and seeds 5-9 on GPU 2, in parallel (each NF4 model fits one 16 GB card)
#   2 bf16 reference: one model split over GPUs 0+2 (16.4 GB of weights), all 10 seeds
#   3 merge the two NF4 halves and pack runs/E26/a4000 into ~/e26_clarity/e26_a4000.tgz
#   nohup setsid bash ~/e26_clarity/repo/code/launch/e26_lab_run.sh > ~/e26_clarity/run.log 2>&1 &
set -uo pipefail
W=$HOME/e26_clarity
R=$W/repo
GPUS=(0 2)
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*"; }
echo $$ > $W/run.pid

free_gpus() {  # every listed GPU must have no compute process and < 1 GB in use (ours included)
  local g n used
  for g in "$@"; do
    n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" -ne 0 ] || [ "$used" -ge 1000 ]; then log "GPU $g busy ($n processes, $used MiB)"; return 1; fi
  done
}

log "phase 0: setup in $W"
# no python3-venv on this server (and no sudo): uv, with its binary, Python, cache and venv all inside $W and no
# change to the shell profile
export UV_INSTALL_DIR=$W/bin UV_NO_MODIFY_PATH=1 UV_CACHE_DIR=$W/uvcache UV_PYTHON_INSTALL_DIR=$W/uvpython
[ -x $W/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh || { log "uv install failed"; exit 1; }
if [ ! -x $W/venv/bin/python ]; then
  $W/bin/uv venv $W/venv --python /usr/bin/python3 --no-python-downloads || { log "venv failed"; exit 1; }  # the system 3.10: fetching a managed Python stalled on this network
  $W/bin/uv pip install -q --python $W/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124 \
    || { log "torch install failed"; exit 1; }
  $W/bin/uv pip install -q --python $W/venv/bin/python "transformers==5.17.0" "peft==0.20.0" "accelerate==1.15.0" \
    bitsandbytes pandas pyarrow "huggingface_hub>=1.28" python-dotenv scipy scikit-learn || { log "package install failed"; exit 1; }
fi
PY=$W/venv/bin/python
$PY -c "import torch, bitsandbytes, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)"
cd $R/code
$PY - <<'EOF' || { echo "download failed"; exit 1; }
from huggingface_hub import snapshot_download
pat = ["Q8_fullq_lora_12ep/seed*/*.npy", "Q8_fullq_lora_12ep/seed*/*.json", "Q8_fullq_lora_12ep/seed*/adapter_best/*"]
print(snapshot_download("siddarthg44/clarity-semeval26", local_dir="../runs", allow_patterns=pat))
print(snapshot_download("Qwen/Qwen3-8B-Base"))
EOF
log "phase 0 done"

log "phase 1: NF4"
if free_gpus "${GPUS[@]}"; then
  CUDA_VISIBLE_DEVICES=0 $PY -m audit.quant --precision nf4 --site a4000 --smoke > $W/smoke_nf4.log 2>&1 \
    && grep -E "smoke|Error" $W/smoke_nf4.log | tail -2
  ( CUDA_VISIBLE_DEVICES=0 $PY -m audit.quant --precision nf4 --site a4000_h0 --seeds 0-4 --batch 16 > $W/nf4_h0.log 2>&1 ) &
  P0=$!
  ( CUDA_VISIBLE_DEVICES=2 $PY -m audit.quant --precision nf4 --site a4000_h1 --seeds 5-9 --batch 16 > $W/nf4_h1.log 2>&1 ) &
  P1=$!
  echo "$P0 $P1" > $W/nf4.pids
  wait $P0 $P1
  log "phase 1 done: $(tail -1 $W/nf4_h0.log) / $(tail -1 $W/nf4_h1.log)"
else
  log "phase 1 SKIPPED: GPUs 0/2 not free"
fi

log "phase 2: bf16 reference over GPUs 0+2"
if free_gpus "${GPUS[@]}"; then
  CUDA_VISIBLE_DEVICES=0,2 $PY -m audit.quant --precision bf16 --site a4000 --device-map auto --batch 8 > $W/bf16.log 2>&1
  log "phase 2 done: $(tail -1 $W/bf16.log)"
else
  log "phase 2 SKIPPED: GPUs 0/2 not free"
fi

log "phase 3: merge and pack"
$PY - <<'EOF'
import json, shutil
from pathlib import Path
E = Path("../runs/E26")
dst = E / "a4000" / "nf4"
dst.mkdir(parents=True, exist_ok=True)
rep = {}
for h in ("a4000_h0", "a4000_h1"):
    src = E / h / "nf4"
    if not src.exists():
        continue
    for d in src.glob("seed*"):
        shutil.copytree(d, dst / d.name, dirs_exist_ok=True)
    r = json.loads((src / "report.json").read_text())
    rep.update({k: v for k, v in r.items() if not k.startswith("seed")})
    rep.update({k: v for k, v in r.items() if k.startswith("seed")})
(dst / "report.json").write_text(json.dumps(rep, indent=1))
print("merged NF4 seeds:", sorted(p.name for p in dst.glob("seed*")))
EOF
cd $R && tar czf $W/e26_a4000.tgz runs/E26/a4000 && ls -lh $W/e26_a4000.tgz
cp $W/*.log $W/run.log $R/runs/E26/ 2>/dev/null
log "ALL DONE"
