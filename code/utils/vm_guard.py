#!/usr/bin/env python3
"""A rented VM that looks after itself: progress logs on Hugging Face, checked uploads, and it pauses or destroys
itself, so nothing depends on the laptop that launched it (2026-10-29).

    python -m utils.vm_guard heartbeat --dest logs/infer_lane FILE...    # copy log files to the HF repo
    python -m utils.vm_guard upload --dest E21 DIR [--allow PATTERN...]  # upload a folder
    python -m utils.vm_guard verify PATH...                              # exit 1 unless every path is on HF
    python -m utils.vm_guard stop {pause,destroy}                        # stop THIS instance

HF_TOKEN and HF_REPO_ID come from code/.env; JL_API_KEY and JL_MACHINE_ID from the environment (the launcher
writes them to /home/clarity_llm/keys.env). Every command retries network errors before giving up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def retry(fn, tries: int = 5, wait: float = 20.0):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- network errors of any kind are retried
            print(f"[vm_guard] attempt {k + 1}/{tries} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if k + 1 == tries:
                raise
            time.sleep(wait)


def hf():
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / "code" / ".env")
    return HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO_ID"]


def heartbeat(dest: str, files: list[str]) -> None:
    from huggingface_hub import CommitOperationAdd

    api, repo = hf()
    ops = [CommitOperationAdd(path_in_repo=f"{dest}/{Path(f).name}", path_or_fileobj=f) for f in files if Path(f).is_file()]
    if ops:
        retry(lambda: api.create_commit(repo_id=repo, operations=ops, commit_message=f"heartbeat: {dest}"))
        print(f"[vm_guard] heartbeat: {len(ops)} files -> {dest}", flush=True)


def upload(dest: str, folder: str, allow: list[str] | None) -> None:
    api, repo = hf()
    retry(lambda: api.upload_folder(folder_path=folder, path_in_repo=dest, repo_id=repo, allow_patterns=allow,
                                    commit_message=f"upload: {dest}"))
    print(f"[vm_guard] uploaded {folder} -> {dest}", flush=True)


def verify(paths: list[str]) -> bool:
    api, repo = hf()
    files = set(retry(lambda: api.list_repo_files(repo)))
    missing = [p for p in paths if p not in files and not any(f.startswith(p.rstrip("/") + "/") for f in files)]
    print(f"[vm_guard] verify: {len(paths) - len(missing)}/{len(paths)} on the Hub"
          + (f"; missing {missing}" if missing else ""), flush=True)
    return not missing


def stop(action: str) -> None:
    from jarvislabs import Client

    mid = int(os.environ["JL_MACHINE_ID"])
    c = Client()
    print(f"[vm_guard] {action} instance {mid}", flush=True)
    retry(lambda: getattr(c.instances, action)(mid), tries=10, wait=30)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heartbeat"); h.add_argument("--dest", required=True); h.add_argument("files", nargs="+")
    u = sub.add_parser("upload"); u.add_argument("--dest", required=True); u.add_argument("folder")
    u.add_argument("--allow", nargs="*", default=None)
    v = sub.add_parser("verify"); v.add_argument("paths", nargs="+")
    s = sub.add_parser("stop"); s.add_argument("action", choices=("pause", "destroy"))
    a = p.parse_args()
    if a.cmd == "heartbeat":
        heartbeat(a.dest, a.files)
    elif a.cmd == "upload":
        upload(a.dest, a.folder, a.allow)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.paths) else 1)
    else:
        stop(a.action)


if __name__ == "__main__":
    main()
