#!/usr/bin/env python3
"""Experiment tracking: Weights & Biases metrics and Hugging Face Hub artifacts.

Credentials come from `clarity/.env` (see `.env.example`). The file is re-read at
the start of EVERY run, so keys added while a pipeline is already going take
effect from the next run onward -- the pipeline does not need restarting.

Nothing here is allowed to take a training run down:

  * no WANDB_API_KEY  -> W&B runs in OFFLINE mode and writes under clarity/wandb/.
                         Upload those later with:  wandb sync clarity/wandb/offline-run-*
  * no HF_TOKEN / HF_REPO_ID -> the upload is skipped with a message. Weights are
                         still saved locally, and can be pushed later with:
                         python tracking.py push runs/<config name>
  * any exception from either service is caught and printed, never raised.

CLI:
    python tracking.py push runs/L0_large_base        # push every seed dir
    python tracking.py push runs/L0_large_base/seed0  # push one
    python tracking.py push-pending runs [NAME ...]   # queue background uploads of unuploaded runs
    python tracking.py push-pending-sync runs         # same, but blocking (manual use)
    python tracking.py relog runs/L0_large_base/seed0 # rebuild a W&B run from metrics.json

A successful upload writes `.hf_pushed` in the run directory; `push-pending`
uploads every finished run (has metrics.json) without that marker, so a failed or
interrupted upload is retried on the next pipeline pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Uploads go through classic LFS, not the Xet backend. Measured on this server
# 2026-09-19: Xet chunked ~225 MB of the checkpoint locally and then sent 0 bytes
# for 14+ minutes on two open connections; with Xet disabled the same 870 MB file
# uploaded in 38 s (~23 MB/s). Set before huggingface_hub is imported, which in
# the upload worker (this file, run as a script) happens inside push_run.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
DEFAULT_PROJECT = "clarity-semeval26"


def load_env() -> None:
    """Load clarity/.env; its values win over anything already in the environment."""
    try:
        from dotenv import load_dotenv

        # override=True: .env is the source of truth. With override=False, a process that
        # started before the keys were filled kept the template values (this is how
        # L0_large_base seed0 tried to upload to "your-hf-username").
        load_dotenv(ENV_FILE, override=True)
    except Exception as e:  # pragma: no cover - dotenv is installed; be defensive anyway
        print(f"[env] could not read {ENV_FILE}: {e}")
    # An empty value in .env means "not set", not "set to the empty string".
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN", "HF_REPO_ID"):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)


class Tracker:
    """Thin W&B wrapper whose every method is a safe no-op on failure."""

    def __init__(self, *, name: str, seed: int | str, group: str, job_type: str, config: dict,
                 state_dir: Path | None = None):
        """`state_dir`: if given, the W&B run id is kept in `<state_dir>/wandb_id.txt`,
        so a run resumed after a restart continues the SAME W&B run."""
        load_env()
        self.run = None
        try:
            import wandb

            run_id = None
            if state_dir is not None:
                id_file = Path(state_dir) / "wandb_id.txt"
                if id_file.exists():
                    run_id = id_file.read_text().strip()
                else:
                    # wandb 0.28 has no public id generator; W&B accepts any short
                    # lowercase alphanumeric id.
                    import secrets

                    run_id = secrets.token_hex(4)
                    Path(state_dir).mkdir(parents=True, exist_ok=True)
                    id_file.write_text(run_id)

            mode = os.environ.get("WANDB_MODE") or (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
                entity=os.environ.get("WANDB_ENTITY"),
                name=f"{name}-" + (f"s{seed}" if str(seed).isdigit() else str(seed)),
                group=group,
                job_type=job_type,
                config=config,
                dir=str(ROOT),
                mode=mode,
                id=run_id,
                resume="allow" if run_id else None,
            )
            print(f"[wandb] {mode} run {self.run.name} (group={group})")
        except Exception as e:
            print(f"[wandb] disabled for this run: {type(e).__name__}: {e}")

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
        except Exception as e:
            print(f"[wandb] log failed: {e}")

    def summary(self, metrics: dict) -> None:
        if self.run is None:
            return
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    self.run.summary[k] = v
        except Exception as e:
            print(f"[wandb] summary failed: {e}")

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")


def push_run(outdir: Path, name: str, seed: int | str) -> bool:
    """Upload a run directory to `HF_REPO_ID` under `<name>/seed<k>/`. Private repo."""
    load_env()
    token, repo = os.environ.get("HF_TOKEN"), os.environ.get("HF_REPO_ID")
    if not token or not repo:
        print(
            "[hf] upload skipped: HF_TOKEN and/or HF_REPO_ID not set in clarity/.env. "
            f"Weights are saved locally; push later with: python tracking.py push {outdir}"
        )
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        seed_part = str(seed) if str(seed).startswith(("seed", "fold")) else f"seed{seed}"
        api.upload_folder(
            folder_path=str(outdir),
            ignore_patterns=["resume.pt", "*.tmp", ".hf_pushed", "wandb_id.txt"],
            repo_id=repo,
            path_in_repo=f"{name}/{seed_part}",
            commit_message=f"{name} {seed_part}",
        )
        (Path(outdir) / ".hf_pushed").write_text(f"{repo}/{name}/{seed_part}\n")
        print(f"[hf] uploaded {outdir} -> {repo}/{name}/{seed_part}")
        return True
    except Exception as e:
        print(f"[hf] upload failed (run is unaffected): {type(e).__name__}: {e}")
        return False


UPLOAD_LOG = ROOT / "logs" / "hf_uploads.log"
UPLOAD_LOCK = ROOT / "logs" / ".hf_upload.lock"
UPLOAD_TIMEOUT_S = 3600


def push_async(outdir: Path) -> None:
    """Upload in a DETACHED background process and return immediately.

    Uploads must never hold the GPU: a stalled connection on 2026-09-19 kept the
    pipeline idle for 14+ minutes while it sat inside a synchronous upload. The
    worker takes a file lock so uploads run one at a time, and is killed after
    UPLOAD_TIMEOUT_S; an upload that fails or times out leaves no .hf_pushed
    marker, so the next `push-pending` retries it.
    """
    import subprocess

    UPLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"flock -w 14400 {UPLOAD_LOCK} timeout {UPLOAD_TIMEOUT_S} "
           f"{sys.executable} {Path(__file__).resolve()} push {Path(outdir).resolve()}")
    with open(UPLOAD_LOG, "a") as log:
        log.write(f"queued: {outdir}\n")
        subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(ROOT))
    print(f"[hf] upload queued in background: {outdir} (progress: {UPLOAD_LOG})")


def pending_runs(runs_root: Path, names: list[str] | None = None) -> list[Path]:
    """Finished seed runs with saved weights that have not been uploaded yet.

    `names` restricts the scan to those configurations. Without it every run under
    `runs_root` is eligible -- which is how a smoke test's runs once got queued for
    upload by a lane that was only meant to retry its own experiment's uploads.
    """
    out = []
    for m in sorted(Path(runs_root).glob("*/seed*/metrics.json")):
        d = m.parent
        if names is not None and d.parent.name not in names:
            continue
        if d.parent.name.startswith("_") or (d / ".hf_pushed").exists():
            continue
        if (d / "model.pt").exists():  # runs saved without weights (e.g. folds) are skipped
            out.append(d)
    return out


def push_pending(runs_root: Path) -> None:
    """Upload every finished seed run that has not been uploaded yet (blocking)."""
    for d in pending_runs(runs_root):
        push_run(d, d.parent.name, d.name)


def relog(run_dir: Path) -> None:
    """Recreate a W&B run from a finished run's metrics.json (per-epoch history)."""
    import json

    run_dir = Path(run_dir).resolve()
    m = json.loads((run_dir / "metrics.json").read_text())
    name, seed = run_dir.parent.name, run_dir.name.replace("seed", "")
    t = Tracker(name=name, seed=seed, group=name, job_type="train", config=m.get("config", {}),
                state_dir=run_dir)
    for row in m.get("history", []):
        t.log(row, step=row["epoch"])
    t.summary({k: v for k, v in m.items() if isinstance(v, (int, float))})
    t.finish()
    print(f"[wandb] relogged {len(m.get('history', []))} epochs for {name} seed {seed}")


def _cli() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("push", "push-pending", "push-pending-sync", "relog"):
        raise SystemExit(__doc__)
    target = Path(sys.argv[2]).resolve()
    names = sys.argv[3:] or None  # optional: only these configurations
    if sys.argv[1] == "push-pending":  # non-blocking: queue background uploads and return
        for d in pending_runs(target, names):
            push_async(d)
        return None
    if sys.argv[1] == "push-pending-sync":
        return push_pending(target)
    if sys.argv[1] == "relog":
        return relog(target)
    dirs = [target] if target.name.startswith(("seed", "fold")) else sorted(target.glob("seed*"))
    for d in dirs:
        push_run(d, d.parent.name, d.name)


if __name__ == "__main__":
    _cli()
