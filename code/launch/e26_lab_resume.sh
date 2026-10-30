#!/usr/bin/env bash
# E26 lab, resume after the first pass ran out of GPU memory part-way (memory crept up across seeds in one
# process). One process per seed (a fresh CUDA context each time); finished seeds are skipped by audit.quant.
# Same rules as e26_lab_run.sh: only GPUs 0 and 2, only if they carry no other process.
set -uo pipefail
W=$HOME/e26_clarity; R=$W/repo; PY=$W/venv/bin/python
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*"; }
echo $$ > $W/run.pid
busy() { local n; n=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$n" -ne 0 ]; }
cd $R/code
log "resume: NF4 seeds 3-4 (GPU 0) and 8-9 (GPU 2)"
if busy 0 || busy 2; then log "GPUs 0/2 not free: stop"; exit 1; fi
( for k in 3 4; do CUDA_VISIBLE_DEVICES=0 $PY -m audit.quant --precision nf4 --site a4000_h0 --seeds $k-$k --batch 16 >> $W/nf4_h0.log 2>&1; done ) &
A=$!
( for k in 8 9; do CUDA_VISIBLE_DEVICES=2 $PY -m audit.quant --precision nf4 --site a4000_h1 --seeds $k-$k --batch 16 >> $W/nf4_h1.log 2>&1; done ) &
B=$!
wait $A $B
log "NF4 seeds: $(ls -d ../runs/E26/a4000_h0/nf4/seed* ../runs/E26/a4000_h1/nf4/seed* 2>/dev/null | wc -l)/10"
log "resume: bf16 seeds 1-9 over GPUs 0+2 (12 GiB per card)"
if busy 0 || busy 2; then log "GPUs 0/2 not free: stop"; exit 1; fi
for k in 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0,2 E26_MAX_MEM_GIB=12 $PY -m audit.quant --precision bf16 --site a4000 --device-map auto --batch 8 \
    --seeds $k-$k >> $W/bf16.log 2>&1 || log "bf16 seed $k failed"
done
log "bf16 seeds: $(ls -d ../runs/E26/a4000/bf16/seed* 2>/dev/null | wc -l)/10"
$PY - <<'PYEOF'
import json, shutil
from pathlib import Path
E = Path("../runs/E26"); dst = E / "a4000" / "nf4"; dst.mkdir(parents=True, exist_ok=True); rep = {}
for h in ("a4000_h0", "a4000_h1"):
    src = E / h / "nf4"
    if src.exists():
        for d in src.glob("seed*"):
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)
        rep.update(json.loads((src / "report.json").read_text()))
(dst / "report.json").write_text(json.dumps(rep, indent=1))
print("merged NF4 seeds:", sorted(p.name for p in dst.glob("seed*")))
PYEOF
cd $R && tar czf $W/e26_a4000.tgz runs/E26/a4000 && cp $W/*.log runs/E26/ 2>/dev/null; ls -lh $W/e26_a4000.tgz
log "ALL DONE"
