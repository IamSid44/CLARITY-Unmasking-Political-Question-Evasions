#!/usr/bin/env bash
# E26, NF4 half (docs/02 E26): run on the local machine with the RTX A4000s (3 x 16 GB). Free; ~1-1.5 h.
#   cd <repo> && bash code/launch/e26_local_nf4.sh            # from the repository root (git pull first)
# Runs, in parallel: the 16-bit reference split over GPUs 0+1, and NF4 on GPU 2 (each: 890 items x 10 seeds).
# At the end it packs runs/E26/a4000 into e26_a4000.tgz in the repository root: copy that file to the laptop's
# repository root and run `tar xzf e26_a4000.tgz && cd code && python -m preregistered_analyses.e26_analysis`.
# Needs: NVIDIA driver + internet to pypi.org and huggingface.co (the adapters are in a public repo).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
ROOT=$PWD
VENV=${VENV:-$ROOT/.venv-e26}
export HF_HOME=${HF_HOME:-$ROOT/.hf} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/e26_local
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee logs/e26_local/gpus.txt

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
  "$VENV/bin/pip" install -q "transformers==5.17.0" "peft==0.20.0" "accelerate==1.15.0" bitsandbytes pandas pyarrow \
    "huggingface_hub>=1.28" python-dotenv scipy scikit-learn
fi
PY=$VENV/bin/python
"$PY" -c "import torch, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')"

# the 10 adapters and their saved probabilities (public Hub repo; no token needed)
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
pat = ["Q8_fullq_lora_12ep/seed*/*.npy", "Q8_fullq_lora_12ep/seed*/*.json", "Q8_fullq_lora_12ep/seed*/adapter_best/*"]
print(snapshot_download("siddarthg44/clarity-semeval26", local_dir="runs", allow_patterns=pat))
EOF
cd code
"$PY" -m audit.frames build
"$PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B-Base'))"

CUDA_VISIBLE_DEVICES=2 "$PY" -m audit.quant --precision nf4 --site a4000 --smoke 2>&1 | tee ../logs/e26_local/smoke_nf4.log
( CUDA_VISIBLE_DEVICES=0,1 "$PY" -m audit.quant --precision bf16 --site a4000 --device-map auto --batch 8 \
    > ../logs/e26_local/bf16.log 2>&1 ) &
B=$!
CUDA_VISIBLE_DEVICES=2 "$PY" -m audit.quant --precision nf4 --site a4000 --batch 16 > ../logs/e26_local/nf4.log 2>&1
wait $B
tail -2 ../logs/e26_local/bf16.log ../logs/e26_local/nf4.log
cd "$ROOT" && tar czf e26_a4000.tgz runs/E26/a4000 logs/e26_local && ls -lh e26_a4000.tgz
echo "copy e26_a4000.tgz to the laptop's repository root"
