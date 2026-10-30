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
#
# The lane looks after itself, so the laptop can disconnect (2026-10-29):
#   * a failed step is recorded and the lane goes on with every step that does not depend on it;
#   * every HEARTBEAT_S seconds, logs/infer*.log are copied to HF (logs/infer_lane/), readable in a browser;
#   * a watchdog stops the lane if no log or output file changes for IDLE_MIN minutes, or after MAX_HOURS;
#   * at the end, or on any exit: upload the outputs and logs, check they are on HF, then DESTROY the instance
#     if every step succeeded, otherwise PAUSE it (the disk is kept; only storage is billed).
#     Needs JL_API_KEY and JL_MACHINE_ID in $BASE/keys.env (`jarvis_drive.sh lane` writes them).
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
IDLE_MIN=${IDLE_MIN:-45}
MAX_HOURS=${MAX_HOURS:-4}
HEARTBEAT_S=${HEARTBEAT_S:-600}
AUTO_STOP=${AUTO_STOP:-1}  # 0: never pause or destroy the instance (debugging only)
PASS=${PASS:-1}  # 2: the second 27B pass (cascade.pass2_jobs; E19-E system part, E22c ladder, E21b normalized test)
                 # 3: E25 (question swaps, answer prefixes, frozen-model probes, the LLM evasion audit; audit/)
KEYS=$BASE/keys.env
SELF=$(readlink -f "$0")
STEPS="deps fetch m1inputs jobs normalize e21 specificity vllm_dev faith vllm_cached vllm_orders vllm_rescore vllm_all9 vllm_think m1cost, then upload"

export HF_HOME=$BASE/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export PATH="$HOME/.local/bin:$PATH"  # uv, installed by jarvis_setup.sh setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() { mkdir -p "$LOGS"; echo "$(date '+%F %T') $*" | tee -a "$LOGS/infer.log"; }
die()   { stage "FAILED: $*"; exit 1; }
run()   { cd "$CODE/code" && "$@"; }
step()  {  # step <name> <command...>: once, logged, with a done-flag; a failure is recorded, not fatal
  local name=$1; shift
  [ -f "$LOGS/.infer_$name" ] && { stage "$name: already done"; return 0; }
  stage "$name: start"
  if "$@" > "$LOGS/infer_$name.log" 2>&1; then touch "$LOGS/.infer_$name"; stage "$name: done ($(tail -1 "$LOGS/infer_$name.log" | cut -c1-150))"
  else
    echo "$name" >> "$LOGS/.infer_failed"
    stage "FAILED: $name (logs/infer_$name.log: $(tail -2 "$LOGS/infer_$name.log" | tr '\n' ' ' | cut -c1-200))"
    return 1
  fi
}
ok()    { [ -f "$LOGS/.infer_$1" ]; }
guard() { (cd "$CODE/code" && set -a && { [ -f "$KEYS" ] && . "$KEYS"; true; } && set +a && "$PY" -m utils.vm_guard "$@"); }

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
  # Triton compiles a C helper when vLLM starts, which needs the Python headers
  dpkg -s python3.12-dev >/dev/null 2>&1 || sudo apt-get install -y -qq python3.12-dev >/dev/null
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

adj() {  # vLLM JIT-compiles kernels at start-up: it needs the venv's ninja and the CUDA toolkit's nvcc on PATH
  PATH="$BASE/venv-vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda     run "$VPY" -m cascade.adjudicate --backend vllm --model "$M3_MODEL" --k 0 "$@"
}

wait_bg() {  # the background 27B download and vLLM install must have finished
  for f in fetch27 vllm_env; do
    while [ ! -f "$LOGS/.infer_$f" ]; do
      [ -f "$LOGS/.infer_${f}_failed" ] && { echo "$f" >> "$LOGS/.infer_failed"; stage "FAILED: $f (logs/infer_$f.log)"; return 1; }
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

vllm_rescore() {  # E19 amendment: the faith gate failed (0.925 < 0.98), so every saved record set is re-scored here
  run "$PY" - <<'EOF' || return 1
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
load_dotenv(".env")
for f in ("frontier_jobs.jsonl", "frontier_jobs_Q8_distill_a05.jsonl"):
    print(hf_hub_download(os.environ["HF_REPO_ID"], f"M3/{f}", local_dir="../runs", token=os.environ["HF_TOKEN"]))
EOF
  cat "$CODE/runs/M3/frontier_jobs.jsonl" "$CODE/runs/M3/frontier_jobs_Q8_distill_a05.jsonl" > "$CODE/runs/E22/jobs_frontier.jsonl"
  # the VM's dev records (tag qwen38_27b_k0_jarvis) are copied in, so dev is not scored twice
  mkdir -p "$CODE/runs/M3/qwen38_27b_k0_vm"
  [ -s "$CODE/runs/M3/qwen38_27b_k0_vm/responses.jsonl" ] ||     cp "$CODE/runs/M3/qwen38_27b_k0_jarvis/responses.jsonl" "$CODE/runs/M3/qwen38_27b_k0_vm/responses.jsonl"
  adj --tag qwen38_27b_k0_vm --splits dev,test,screen,slice,trainall --jobs-file ../runs/E22/jobs_frontier.jsonl
}

hf_fetch2() {  # pass 2: M1's cascade inputs, the embeddings, the VM's record sets, and the pass-2 jobs
  run "$PY" - <<'EOF'
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(".env")
pat = ["M1/*", "M3/embeddings/*", "E22/jobs_p2_*.jsonl", "E22/pass2_subsets.json"]
pat += [f"M3/qwen38_27b_k0_{t}/*" for t in ("vm", "shift1", "shift2", "shift3")]
print(snapshot_download(os.environ["HF_REPO_ID"], local_dir="../runs", allow_patterns=pat, token=os.environ["HF_TOKEN"]))
EOF
}

p2_students() {  # the students' own top 4 in the three shifted orders (permutation-averaged q, E19 arm B adopted)
  for o in shift1 shift2 shift3; do
    adj --tag "qwen38_27b_k0_$o" --splits "" --jobs-file ../runs/E22/jobs_p2_students.jsonl --order "$o" || return 1
  done
}

p2_testnorm() {  # E21b: normalized official test answers, all four cyclic orders
  local o tag
  for o in taxonomy shift1 shift2 shift3; do
    tag=qwen38_27b_k0_vm_tn; [ "$o" = taxonomy ] || tag="${tag}_$o"
    adj --tag "$tag" --splits "" --jobs-file ../runs/E22/jobs_p2_testnorm.jsonl --order "$o" --test-text normalized || return 1
  done
}

pass2() {
  step p2_fetch hf_fetch2
  stage "waiting for the 27B download and vLLM install"
  if wait_bg && ok p2_fetch; then
    step p2_dev adj --tag qwen38_27b_k0_vm --splits "" --jobs-file ../runs/E22/jobs_p2_dev.jsonl
    step p2_students p2_students
    step p2_testnorm p2_testnorm
  fi
}

# ---------------------------------------------------------------- PASS=3: E25 (docs/02 E25 registration)
SUBJECTS="llama31_8b=meta-llama/Llama-3.1-8B-Instruct mistral_7b=mistralai/Mistral-7B-Instruct-v0.3 qwen3_8b=Qwen/Qwen3-8B olmo2_7b=allenai/OLMo-2-1124-7B-Instruct"

fetch_subjects() {  # the four answering LLMs, safetensors only (no duplicate consolidated/original weights)
  run "$PY" - <<EOF
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(".env")
lines = []
for spec in "$SUBJECTS".split():
    name, hf_id = spec.split("=")
    p = snapshot_download(hf_id, token=os.environ["HF_TOKEN"],
                          ignore_patterns=["original/*", "consolidated*", "*.pth", "*.gguf", "*.bin"])
    print(name, p, flush=True)
    lines.append(f"{name}={p}")
open("$LOGS/subjects.txt", "w").write("\n".join(lines) + "\n")
EOF
}

wait_flag() {  # wait_flag <name>: a background job's done-flag (fails if it failed)
  while [ ! -f "$LOGS/.infer_$1" ]; do
    [ -f "$LOGS/.infer_${1}_failed" ] && { echo "$1" >> "$LOGS/.infer_failed"; stage "FAILED: $1 (logs/infer_$1.log)"; return 1; }
    sleep 30
  done
}

p3_generate() {  # one process per model (vLLM engines do not always release the GPU); a failed model does not stop the rest
  local spec bad=0
  while read -r spec; do
    [ -n "$spec" ] || continue
    PATH="$BASE/venv-vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda run "$VPY" -m audit.generate --model "$spec" \
      < /dev/null || { echo "generation failed: $spec"; bad=1; }
  done < "$LOGS/subjects.txt"
  return $bad
}

p3_27b() { for o in taxonomy shift1 shift2 shift3; do adj --tag "e25_$o" --splits "" --jobs-file ../runs/E25/jobs_27b.jsonl --order "$o" || return 1; done; }

p3_upload() {  # E25 outputs (without the 1 GB train hidden states) and the 27B records
  local root=$CODE/runs
  guard upload --dest E25 "$root/E25" --allow 'frames/*' 'm1/*' 'labels/*' 'hidden/dev_*' 'hidden/test_*' \
    'hidden/latency.json' '*.json' '*.jsonl' '*.npz' || return 1
  for o in taxonomy shift1 shift2 shift3; do
    [ -d "$root/M3/e25_$o" ] && { guard upload --dest "M3/e25_$o" "$root/M3/e25_$o" || return 1; }
  done
  return 0
}

pass3() {
  for f in subjects; do  # background, alongside fetch27 and vllm_env
    [ -f "$LOGS/.infer_$f" ] && continue
    ( fetch_subjects > "$LOGS/infer_$f.log" 2>&1 && touch "$LOGS/.infer_$f" || touch "$LOGS/.infer_${f}_failed" ) &
  done
  step deps uv pip install --python "$PY" scipy scikit-learn
  step fetch hf_fetch
  ok fetch && step m1inputs m1inputs
  step p3_frames run "$PY" -m audit.frames build
  ok fetch && ok p3_frames && step p3_m1swap run "$PY" -m audit.infer --frames 'dev_*,test_*'
  ok p3_frames && step p3_hidden run "$PY" -m audit.hidden dump
  if ok p3_hidden; then  # CPU probes in the background while the GPU generates
    ( cd "$CODE/code" && "$PY" -m audit.hidden probe > "$LOGS/infer_p3_probe.log" 2>&1 \
      && touch "$LOGS/.infer_p3_probe" || touch "$LOGS/.infer_p3_probe_failed" ) &
  fi
  stage "waiting for the vLLM install and the subject models"
  if wait_flag vllm_env && wait_flag subjects; then
    step p3_generate p3_generate
    step p3_llmframe run "$PY" -m audit.frames llm
    ok p3_llmframe && step p3_m1llm run "$PY" -m audit.infer --frames llm
  fi
  step p3_jobs run "$PY" -m audit.system jobs '*'
  stage "waiting for the 27B download"
  if wait_flag fetch27 && ok p3_jobs && ok m1inputs; then
    step p3_27b p3_27b
    step p3_label run "$PY" -m audit.system label '*'
  fi
  ok p3_hidden && { stage "waiting for the probes"; wait_flag p3_probe; }
  rm -f "$CODE/runs/E25/hidden/train.npy"  # 1 GB, not uploaded; the probes are fitted
}

# ---------------------------------------------------------------- PASS=4: E26 (FP8 vs 16-bit, RTX PRO 6000)
p4_27b() { for o in taxonomy shift1 shift2 shift3; do adj --tag "e26_rtxpro_$o" --splits "" --jobs-file ../runs/E26/rtxpro/jobs_27b.jsonl --order "$o" --vllm-args '{"enforce_eager": true}' || return 1; done; }

pass4() {
  step deps uv pip install --python "$PY" scipy scikit-learn
  step fetch hf_fetch
  ok fetch && step m1inputs m1inputs
  step p4_frames run "$PY" -m audit.frames build
  ok fetch && ok p4_frames && step p4_smoke run "$PY" -m audit.quant --precision fp8 --smoke
  ok p4_frames && step p4_bf16 run "$PY" -m audit.quant --precision bf16 --site rtxpro
  ok p4_smoke && step p4_fp8 run "$PY" -m audit.quant --precision fp8 --site rtxpro
  ok p4_bf16 && step p4_jobs run "$PY" -m audit.quant_system jobs --site rtxpro
  stage "waiting for the 27B download and vLLM install"
  if wait_flag fetch27 && wait_flag vllm_env && ok p4_jobs && ok m1inputs; then
    step p4_27b p4_27b
  fi
  ok p4_bf16 && step p4_label run "$PY" -m audit.quant_system label --site rtxpro
}

# ---------------------------------------------------------------- PASS=5: E26 NF4 on the RTX PRO 6000 (bf16 exists)
hf_fetch_m1() {  # only the 10 M1 adapters and their saved probabilities
  run "$PY" -c "
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv('.env')
pat = ['Q8_fullq_lora_12ep/seed*/*.npy', 'Q8_fullq_lora_12ep/seed*/*.json', 'Q8_fullq_lora_12ep/seed*/adapter_best/*']
print(snapshot_download(os.environ['HF_REPO_ID'], local_dir='../runs', allow_patterns=pat, token=os.environ['HF_TOKEN']))"
}

pass5() {
  step deps uv pip install --python "$PY" scipy scikit-learn bitsandbytes
  step p5_fetch hf_fetch_m1
  step p5_frames run "$PY" -m audit.frames build
  ok p5_fetch && ok p5_frames && step p5_smoke run "$PY" -m audit.quant --precision nf4 --smoke
  ok p5_smoke && step p5_nf4 run "$PY" -m audit.quant --precision nf4 --site rtxpro
}

vllm_think() {
  if [ "$(TZ=Asia/Kolkata date +%H:%M)" \> "$THINK_DEADLINE" ]; then echo "past $THINK_DEADLINE: thinking rung dropped (E20 amendment)"; return 0; fi
  adj --tag qwen38_27b_k0_think4k --splits "" --jobs-file ../runs/E22/jobs_think.jsonl --think-budget 4096 --chunk 128
}

m1cost() {  # one M1 seed and the 27B call, timed on this VM (E22a: every cost ratio from one GPU)
  run "$PY" -m cascade.cost --run "$M1_RUN" --seed 3 --m3-tag qwen38_27b_k0_jarvis || return 1
  mkdir -p "$CODE/runs/E22" && cp "$CODE/runs/M3/cost.json" "$CODE/runs/E22/cost_jarvis.json"
}

upload() {  # every output that exists, then the logs; each folder is checked on the Hub afterwards
  local f paths=() root=$CODE/runs
  if [ "$PASS" = 5 ]; then  # E26, NF4 on the same GPU type as the FP8 run
    guard upload --dest E26/rtxpro/nf4 "$root/E26/rtxpro/nf4" || return 1
    guard upload --dest logs/infer_lane_e26_nf4 "$LOGS" --allow 'infer*' || return 1
    guard verify E26/rtxpro/nf4 logs/infer_lane_e26_nf4/infer.log
    return
  fi
  if [ "$PASS" = 4 ]; then  # E26 only
    guard upload --dest E26 "$root/E26" || return 1
    paths+=("E26/rtxpro")
    for f in $(cd "$root" && ls -d M3/e26_* 2>/dev/null); do guard upload --dest "$f" "$root/$f" || return 1; paths+=("$f"); done
    guard upload --dest logs/infer_lane_e26 "$LOGS" --allow 'infer*' || return 1
    guard verify "${paths[@]}" logs/infer_lane_e26/infer.log
    return
  fi
  if [ "$PASS" = 3 ]; then  # E25 only; the earlier passes' folders are already on the Hub
    p3_upload || return 1
    for f in E25/frames/summary.json $(cd "$root" && ls -d M3/e25_* 2>/dev/null); do paths+=("$f"); done
    guard upload --dest logs/infer_lane "$LOGS" --allow 'infer*' 'subjects.txt' || return 1
    guard verify "${paths[@]}" logs/infer_lane/infer.log
    return
  fi
  for f in E21 E22 M1/Q8_fullq_lora_12ep_fixed_a0.05 $(cd "$root" && ls -d M3/qwen38_27b_k0_* 2>/dev/null); do
    [ -d "$root/$f" ] || continue
    guard upload --dest "$f" "$root/$f" || return 1
    paths+=("$f")
  done
  guard upload --dest logs/infer_lane "$LOGS" --allow 'infer*' || return 1
  guard verify "${paths[@]}" logs/infer_lane/infer.log
}

finish() {  # runs once, however the lane ends: upload, check, then destroy (all steps done) or pause (anything failed)
  local reason=$1
  mkdir "$LOGS/.infer_finish_lock" 2>/dev/null || return 0
  stage "finish: $reason"
  rm -f "$LOGS/.infer_upload"
  local uploaded=0
  step upload upload && uploaded=1
  local failed; failed=$(sort -u "$LOGS/.infer_failed" 2>/dev/null | tr '\n' ' ')
  guard heartbeat --dest logs/infer_lane "$LOGS"/infer*.log >/dev/null 2>&1
  if [ "$AUTO_STOP" != 1 ]; then stage "finish: AUTO_STOP=0, the instance keeps running"; return 0; fi
  if [ ! -f "$KEYS" ]; then stage "finish: NO $KEYS -- cannot stop this instance; stop it by hand"; return 1; fi
  if [ "$reason" = done ] && [ -z "$failed" ] && [ "$uploaded" = 1 ]; then
    stage "finish: every step done and on the Hub -> destroying this instance"
    guard heartbeat --dest logs/infer_lane "$LOGS/infer.log" >/dev/null 2>&1
    guard stop destroy
  else
    stage "finish: reason '$reason', failed steps: ${failed:-none}, uploaded: $uploaded -> pausing (disk kept)"
    guard heartbeat --dest logs/infer_lane "$LOGS/infer.log" >/dev/null 2>&1
    guard stop pause
  fi
}

watchdog() {  # heartbeat to HF; stop the lane if nothing changes for IDLE_MIN minutes or it runs past MAX_HOURS
  local lane_pid=$1 t0 last_hb=0 now why=""
  t0=$(date +%s)
  while kill -0 "$lane_pid" 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    if (( now - last_hb >= HEARTBEAT_S )); then
      guard heartbeat --dest logs/infer_lane "$LOGS"/infer*.log >/dev/null 2>&1; last_hb=$now
    fi
    if [ -z "$(find "$LOGS" "$CODE/runs" -type f -mmin -"$IDLE_MIN" -print -quit 2>/dev/null)" ]; then
      why="watchdog: no log or output change for $IDLE_MIN min"
    elif (( now - t0 > $(awk "BEGIN { print int($MAX_HOURS * 3600) }") )); then  # MAX_HOURS may be fractional
      why="watchdog: over MAX_HOURS=$MAX_HOURS"
    fi
    if [ -n "$why" ]; then
      stage "$why -- stopping the lane"
      echo "watchdog" >> "$LOGS/.infer_failed"
      pkill -TERM -P "$lane_pid"; pkill -TERM -f 'cascade\.|audit\.|vllm' ; sleep 20
      kill -TERM "$lane_pid" 2>/dev/null; sleep 20
      finish "$why"
      return
    fi
  done
}

lane() {
  rm -rf "$LOGS/.infer_finish_lock" "$LOGS/.infer_failed"
  trap 'finish "exit (status $?)"' EXIT
  trap 'exit 143' TERM
  stage "lane: start (PASS=$PASS, IDLE_MIN=$IDLE_MIN, MAX_HOURS=$MAX_HOURS, AUTO_STOP=$AUTO_STOP, keys: $([ -f "$KEYS" ] && echo yes || echo NO))"
  watchdog $$ &
  [ -x "$PY" ] && [ -f "$LOGS/.setup_done" ] || { MODEL=Qwen/Qwen3-8B-Base bash "$CODE/code/launch/jarvis_setup.sh" setup \
    && touch "$LOGS/.setup_done"; } || die "setup"
  mkdir -p "$CODE/runs/E21" "$CODE/runs/E22"
  for f in fetch27 vllm_env; do  # background: weights and environment for the 27B (once, even across restarts)
    [ "$PASS" = 5 ] && break  # E26 NF4 needs neither
    [ -f "$LOGS/.infer_$f" ] && continue
    pidf=$LOGS/.infer_${f}_pid
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && continue
    rm -f "$LOGS/.infer_${f}_failed"
    ( if [ $f = fetch27 ]; then fetch_27b; else vllm_env; fi > "$LOGS/infer_$f.log" 2>&1 \
      && touch "$LOGS/.infer_$f" || touch "$LOGS/.infer_${f}_failed" ) &
    echo $! > "$pidf"
  done
  if [ "$PASS" = 2 ] || [ "$PASS" = 3 ] || [ "$PASS" = 4 ] || [ "$PASS" = 5 ]; then
    case "$PASS" in 2) pass2 ;; 3) pass3 ;; 4) pass4 ;; 5) pass5 ;; esac
    stage "lane: steps finished"
    trap - EXIT
    finish done
    return
  fi
  step deps uv pip install --python "$PY" scipy scikit-learn  # cascade.candidates and decision_rules need them
  step fetch hf_fetch
  ok fetch && step m1inputs m1inputs
  ok m1inputs && step jobs jobs_files
  step normalize run "$PY" -m qevasion.normalize check
  ok fetch && step e21 run "$PY" -m cascade.infer_variants
  step specificity run "$PY" -m cascade.specificity
  stage "waiting for the 27B download and vLLM install"
  if wait_bg && ok m1inputs; then
    step vllm_dev adj --tag qwen38_27b_k0_jarvis --splits dev
    ok vllm_dev && step faith faith
    step vllm_cached adj --tag qwen38_27b_k0_jarvis_cached --splits dev --cache-pad
    step vllm_orders vllm_orders
    step vllm_rescore vllm_rescore
    ok jobs && step vllm_all9 adj --tag qwen38_27b_k0_all9 --splits "" --jobs-file ../runs/E22/jobs_all9.jsonl
    ok jobs && step vllm_think vllm_think
    ok vllm_dev && step m1cost m1cost
  fi
  stage "lane: steps finished"
  trap - EXIT
  finish done
}

case "${1:-}" in
  start)
    command -v tmux >/dev/null || sudo apt-get install -y -qq tmux >/dev/null
    pgrep -f "infer_lane.sh _lane" >/dev/null && { echo "already running: tmux attach -t $SESSION"; exit 0; }
    tmux kill-session -t "$SESSION" 2>/dev/null  # a stale session from an earlier lane
    tmux new-session -d -s "$SESSION" "THINK_DEADLINE='$THINK_DEADLINE' IDLE_MIN=$IDLE_MIN MAX_HOURS=$MAX_HOURS AUTO_STOP=$AUTO_STOP PASS=$PASS bash '$SELF' _lane; exec bash"
    echo "started in tmux session $SESSION; tail -f $LOGS/infer.log" ;;
  _lane) lane ;;
  status) tail -n 20 "$LOGS/infer.log" 2>/dev/null; echo "failed steps: $(sort -u "$LOGS/.infer_failed" 2>/dev/null | tr '
' ' ')"
          nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
  fetch) step fetch hf_fetch ;;
  m1inputs) step m1inputs m1inputs ;;
  *) echo "usage: $0 {start|status|fetch|m1inputs}  (the lane runs: $STEPS)"; exit 2 ;;
esac
