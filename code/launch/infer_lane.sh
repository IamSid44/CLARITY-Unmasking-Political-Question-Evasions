#!/usr/bin/env bash
# The E19/E21/E22 inference lane on one rented GPU VM (docs/06_final_push_handover.md §6). Runs on the instance
# after `jarvis_drive.sh push`:
#
#   bash /home/clarity_llm/code/code/launch/infer_lane.sh start     # everything, in tmux session clarity-infer
#   bash /home/clarity_llm/code/code/launch/infer_lane.sh status
#   bash /home/clarity_llm/code/code/launch/infer_lane.sh <step>    # one step: see STEPS below
#
# Every step writes logs/infer_<step>.log and a done-flag, so a rerun skips finished steps. While the 8B steps
# use the GPU, the 27B weights download and the vLLM environment installs in the background.
set -uo pipefail

BASE=${BASE:-/home/clarity_llm}
CODE=$BASE/code
PY=$BASE/venv/bin/python
VPY=$BASE/venv-vllm/bin/python
LOGS=$BASE/logs
SESSION=clarity-infer
M3_MODEL=Qwen/Qwen3.8-27B-FP8
M1_RUN=Q8_fullq_lora_12ep
THINK_DEADLINE=${THINK_DEADLINE:-"18:00"}  # IST; the thinking rung is dropped after this (E20 amendment)
SELF=$(readlink -f "$0")
STEPS="fetch m1inputs jobs normalize e21 specificity vllm_dev faith vllm_cached vllm_orders vllm_all9 vllm_think m1cost upload"

export HF_HOME=$BASE/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() { mkdir -p "$LOGS"; echo "$(date '+%F %T') $*" | tee -a "$LOGS/infer.log"; }
die()   { stage "FAILED: $*"; exit 1; }
run()   { cd "$CODE/code" && "$@"; }
step()  {  # step <name> <command...>: once, logged, with a done-flag
  local name=$1; shift
  [ -f "$LOGS/.infer_$name" ] && { stage "$name: already done"; return 0; }
  stage "$name: start"
  if "$@" > "$LOGS/infer_$name.log" 2>&1; then touch "$LOGS/.infer_$name"; stage "$name: done ($(tail -1 "$LOGS/infer_$name.log" | cut -c1-150))"
  else die "$name (logs/infer_$name.log: $(tail -2 "$LOGS/infer_$name.log" | tr '\n' ' ' | cut -c1-200))"; fi
}

hf_fetch() {  # the published inputs: seed probabilities and adapters, cross-fit folds, M3 records, embeddings
  run "$PY" - <<'EOF'
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(".env")
pat = []
for run in ("Q8_fullq_lora_12ep", "Q8_distill_a05"):
    pat += [f"{run}/seed*/*.npy", f"{run}/seed*/*.json", f"{run}/seed*/adapter_best/*"]
pat += ["Q8_12ep_crossfit/fold*/*.npy", "Q8_12ep_crossfit/fold*/*.json",
        "M3/qwen38_27b_k0/*", "M3/embeddings/*", "M3/cost.json"]
p = snapshot_download(os.environ["HF_REPO_ID"], local_dir="../runs", allow_patterns=pat, token=os.environ["HF_TOKEN"])
print("fetched into", p)
EOF
}

fetch_27b() { "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$M3_MODEL'))"; }

vllm_env() {
  export PATH="$HOME/.local/bin:$PATH"
  [ -x "$VPY" ] || uv venv "$BASE/venv-vllm" --python 3.12
  uv pip install --python "$VPY" "vllm==0.31.0" pandas pyarrow huggingface_hub python-dotenv
  "$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
}

m1inputs() {  # rebuild M1's cascade inputs and check them against the candidates the saved M3 records used
  run "$PY" -m cascade.candidates --run "$M1_RUN" --crossfit Q8_12ep_crossfit --method fixed || return 1
  run "$PY" - <<'EOF'
import json
import numpy as np
from cascade.adjudicate import M1_INPUTS, select
z = np.load(M1_INPUTS)
saved = {}
for line in open("../runs/M3/qwen38_27b_k0/responses.jsonl"):
    r = json.loads(line)
    if r["split"] in ("dev", "test", "screen"):
        saved[(r["split"], r["i"])] = r["cand"]
bad = {}
for split in ("dev", "test", "screen"):
    jobs = select(split, z)
    bad[split] = sum(saved.get((split, j.i)) != j.cand for j in jobs)
    print(split, len(jobs), "items; candidate sets differing from the saved M3 records:", bad[split])
assert not any(bad.values()), "rebuilt cascade inputs differ from the ones M3 was scored with"
print("cascade inputs reproduce the saved candidate sets")
EOF
}

jobs_files() {  # E19 arm A (all 9 labels) and E20 R2 (thinking within the top 4, singleton train rows)
  run "$PY" - <<'EOF'
import json
import numpy as np
from cascade.adjudicate import M1_INPUTS, topk
from cascade.exemplars import group_keys, load_frames
z = np.load(M1_INPUTS)
fr = load_frames()
with open("../runs/E22/jobs_all9.jsonl", "w") as f:
    for frame in ("dev", "test"):
        for i in range(len(fr[frame])):
            f.write(json.dumps({"split": "all9", "frame": frame, "i": i, "cand": list(range(9))}) + "\n")
ans, _ = group_keys(fr["train"])
_, inv, cnt = np.unique(ans.astype(str), return_inverse=True, return_counts=True)
single = cnt[inv] == 1
q = z["train_q"]; u = 1 - q.max(1); c = topk(q)
edges = np.quantile(u, [0.2, 0.4, 0.6, 0.8])
quint = np.searchsorted(edges, u, side="right")
rng = np.random.default_rng(20)
rows = np.sort(np.concatenate([rng.choice(np.where(single & (quint == k))[0], 100, replace=False) for k in range(5)]))
with open("../runs/E22/jobs_think.jsonl", "w") as f:
    for i in rows:
        f.write(json.dumps({"split": "think500", "frame": "train", "i": int(i), "cand": sorted(map(int, c[i]))}) + "\n")
    for frame in ("dev", "test"):
        for i, ci in enumerate(topk(z[f"{frame}_q"])):
            f.write(json.dumps({"split": f"think_{frame}", "frame": frame, "i": i, "cand": sorted(map(int, ci))}) + "\n")
print(f"singleton train rows {single.sum()} of {len(single)}; think jobs {len(rows)} train + dev + test; all9 jobs written")
EOF
}

adj() { run "$VPY" -m cascade.adjudicate --backend vllm --model "$M3_MODEL" --k 0 "$@"; }

wait_bg() {  # the background 27B download and vLLM install must have finished
  for f in fetch27 vllm_env; do
    while [ ! -f "$LOGS/.infer_$f" ]; do
      [ -f "$LOGS/.infer_${f}_failed" ] && die "$f failed (logs/infer_$f.log)"
      sleep 30
    done
  done
}

faith() {  # E19 amendment: the VM's 27B against the lab server's saved dev records (gate: argmax agreement >= 0.98)
  run "$PY" - <<'EOF'
import json
import numpy as np
def load(tag):
    out = {}
    for line in open(f"../runs/M3/{tag}/responses.jsonl"):
        r = json.loads(line)
        if r["split"] == "dev":
            out[r["i"]] = (tuple(r["cand"]), np.array(r["q"]))
    return out
a, b = load("qwen38_27b_k0"), load("qwen38_27b_k0_jarvis")
both = [i for i in a if i in b and a[i][0] == b[i][0]]
agree = np.mean([a[i][1].argmax() == b[i][1].argmax() for i in both])
dq = np.mean([np.abs(a[i][1] - b[i][1]).max() for i in both])
print(f"faithfulness on {len(both)} dev items: argmax agreement {agree:.3f}, mean max |dq| {dq:.4f}; "
      f"gate (>= 0.98) {'PASSED' if agree >= 0.98 else 'FAILED: re-score saved records on this VM before mixing'}")
EOF
}

vllm_orders() { for o in shift1 shift2 shift3 reverse; do adj --tag "qwen38_27b_k0_$o" --splits dev,test,screen --order "$o" || return 1; done; }

vllm_think() {
  if [ "$(TZ=Asia/Kolkata date +%H:%M)" \> "$THINK_DEADLINE" ]; then echo "past $THINK_DEADLINE: thinking rung dropped (E20 amendment)"; return 0; fi
  adj --tag qwen38_27b_k0_think4k --splits "" --jobs-file ../runs/E22/jobs_think.jsonl --think-budget 4096 --chunk 128
}

m1cost() {  # one M1 seed and the 27B call, timed on this VM (E22a: every cost ratio from one GPU)
  run "$PY" -m cascade.cost --run "$M1_RUN" --seed 3 --m3-tag qwen38_27b_k0_jarvis || return 1
  mkdir -p "$CODE/runs/E22" && cp "$CODE/runs/M3/cost.json" "$CODE/runs/E22/cost_jarvis.json"
}

upload() {
  run "$PY" - <<'EOF'
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
load_dotenv(".env")
api = HfApi(token=os.environ["HF_TOKEN"])
root = Path("../runs")
folders = ["E21", "E22", "M1/Q8_fullq_lora_12ep_fixed_a0.05"] + [f"M3/{p.name}" for p in (root / "M3").iterdir()
                                                                 if p.is_dir() and p.name.startswith("qwen38_27b_k0_")]
for f in folders:
    if (root / f).exists():
        api.upload_folder(folder_path=str(root / f), path_in_repo=f, repo_id=os.environ["HF_REPO_ID"],
                          commit_message=f"inference lane: {f}")
        print("uploaded", f)
api.upload_folder(folder_path="../../logs", path_in_repo="logs/infer_lane", repo_id=os.environ["HF_REPO_ID"],
                  allow_patterns=["infer*"], commit_message="inference lane: logs")
print("uploaded logs")
EOF
}

lane() {
  stage "lane: start"
  [ -x "$PY" ] && [ -f "$LOGS/.setup_done" ] || { MODEL=Qwen/Qwen3-8B-Base bash "$CODE/code/launch/jarvis_setup.sh" setup \
    && touch "$LOGS/.setup_done"; } || die "setup"
  mkdir -p "$CODE/runs/E21" "$CODE/runs/E22"
  for f in fetch27 vllm_env; do  # background: weights and environment for the 27B
    [ -f "$LOGS/.infer_$f" ] && continue
    ( if [ $f = fetch27 ]; then fetch_27b; else vllm_env; fi > "$LOGS/infer_$f.log" 2>&1 \
      && touch "$LOGS/.infer_$f" || touch "$LOGS/.infer_${f}_failed" ) &
  done
  step fetch hf_fetch
  step m1inputs m1inputs
  step jobs jobs_files
  step normalize run "$PY" -m qevasion.normalize check
  step e21 run "$PY" -m cascade.infer_variants
  step specificity run "$PY" -m cascade.specificity
  stage "waiting for the 27B download and vLLM install"; wait_bg
  step vllm_dev adj --tag qwen38_27b_k0_jarvis --splits dev
  step faith faith
  step vllm_cached adj --tag qwen38_27b_k0_jarvis_cached --splits dev --cache-pad
  step vllm_orders vllm_orders
  step vllm_all9 adj --tag qwen38_27b_k0_all9 --splits "" --jobs-file ../runs/E22/jobs_all9.jsonl
  step vllm_think vllm_think
  step m1cost m1cost
  step upload upload
  stage "lane: ALL DONE"
}

case "${1:-}" in
  start)
    command -v tmux >/dev/null || sudo apt-get install -y -qq tmux >/dev/null
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "already running: tmux attach -t $SESSION"; exit 0; }
    tmux new-session -d -s "$SESSION" "THINK_DEADLINE='$THINK_DEADLINE' bash '$SELF' _lane; exec bash"
    echo "started in tmux session $SESSION; tail -f $LOGS/infer.log" ;;
  _lane) lane ;;
  status) tail -n 20 "$LOGS/infer.log" 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
  fetch) step fetch hf_fetch ;;
  m1inputs) step m1inputs m1inputs ;;
  *) echo "usage: $0 {start|status|fetch|m1inputs}  (the lane runs: $STEPS)"; exit 2 ;;
esac
