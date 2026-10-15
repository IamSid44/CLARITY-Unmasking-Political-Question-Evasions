#!/usr/bin/env python3
"""Publish cascade outputs that are not training runs: HF (raw outputs) and W&B (result tables and files).

    python -m cascade.publish hf M3 M4                # runs/M3, runs/M4 -> HF_REPO_ID under M3/, M4/
    python -m cascade.publish wandb E16               # docs/raw/E16_* (+ frontier, cost tables) -> a W&B run

Training runs are published by run_queue.sh (`@hf-push`) and llm_classifier (`--track`); this covers the rest,
so every result is in logs/, on HF and on W&B (the project's rule, 2026-10-14). Uploads share the queue's lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os

from qevasion.paths import FIGURES, LOGS, RAW, RUNS
from utils.tracking import DEFAULT_PROJECT, UPLOAD_LOCK, load_env


def hf(dirs: list[str]) -> None:
    from huggingface_hub import HfApi

    load_env()
    token, repo = os.environ.get("HF_TOKEN"), os.environ.get("HF_REPO_ID")
    if not token or not repo:
        raise SystemExit("HF_TOKEN / HF_REPO_ID not set in code/.env")
    api = HfApi(token=token)
    with open(UPLOAD_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for d in dirs:
            api.upload_folder(folder_path=str(RUNS / d), repo_id=repo, path_in_repo=d, commit_message=f"{d} outputs",
                              ignore_patterns=["*.tmp", "*.lock"])
            msg = f"[hf] uploaded runs/{d} -> {repo}/{d}"
            print(msg)
            with open(LOGS / "hf_uploads.log", "a") as f:
                f.write(msg + "\n")


def wandb_log(exp: str) -> None:
    import wandb

    load_env()
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT), entity=os.environ.get("WANDB_ENTITY"),
                     name=f"{exp}-results", group="cascade", job_type="analysis", config={"experiment": exp})
    art = wandb.Artifact(f"{exp}-results", type="results")
    for f in sorted(RAW.glob(f"{exp}_*.txt")):
        art.add_file(str(f))
    for name in [f.name for f in sorted((RUNS / "M3").glob("frontier_*.json"))] + ["cost.json", "api_ledger.jsonl"]:
        if (RUNS / "M3" / name).exists():
            art.add_file(str(RUNS / "M3" / name))
    for f in sorted(FIGURES.glob(f"{exp}_*.p*")):
        art.add_file(str(f))
    run.log_artifact(art)
    for f in sorted(FIGURES.glob(f"{exp}_*.png")):
        run.log({f.stem: wandb.Image(str(f))})
    for name in [f.name for f in sorted((RUNS / "M3").glob("frontier_*.json"))]:
        p = RUNS / "M3" / name
        if exp in ("E16", "E17", "E18") and "screen" not in name:
            rows = json.loads(p.read_text())["table"]
            cols = list(rows[0])
            run.log({name.replace(".json", ""): wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
    if (RUNS / "M3" / "cost.json").exists():
        run.summary.update({f"cost/{k}": v for k, v in json.loads((RUNS / "M3" / "cost.json").read_text()).items()
                            if isinstance(v, (int, float))})
    print(f"[wandb] {exp}: {run.url}")
    run.finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("hf", "wandb"))
    p.add_argument("targets", nargs="+")
    a = p.parse_args()
    if a.cmd == "hf":
        hf(a.targets)
    else:
        for e in a.targets:
            wandb_log(e)


if __name__ == "__main__":
    main()
