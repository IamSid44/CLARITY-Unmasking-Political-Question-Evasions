"""Epoch-level resume checkpoints, so a server restart costs at most one epoch.

`resume.pt` in a run directory holds everything needed to continue training as
if it had never stopped: model, optimizer and scheduler state, every RNG that
affects the run (python, numpy, torch CPU + CUDA, and the DataLoader's shuffle
generator), plus whatever the caller passes in `extra` -- the per-epoch history,
the best-so-far epoch and its saved predictions.

Writes are atomic (write to a temp file, fsync, rename), so a crash mid-save
leaves the previous checkpoint intact rather than a truncated file.

The checkpoint is deleted once the run writes its metrics.json; it is only for
runs in flight. At ~5 GB for DeBERTa-v3-large (fp32 weights + two AdamW moment
buffers), only one exists at a time because runs are sequential.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

RESUME = "resume.pt"


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_resume(outdir: Path, *, epoch: int, model, opt, sched, gen: torch.Generator, extra: dict) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,  # the LAST COMPLETED epoch
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "gen": gen.get_state(),
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "extra": extra,
        },
        outdir / RESUME,
    )


def load_resume(outdir: Path, *, model, opt, sched, gen: torch.Generator):
    """Restore in place. Returns (next_epoch, extra), or (0, None) if no checkpoint."""
    path = outdir / RESUME
    if not path.exists():
        return 0, None
    st = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    gen.set_state(st["gen"])
    random.setstate(st["rng_py"])
    np.random.set_state(st["rng_np"])
    torch.set_rng_state(st["rng_torch"])
    torch.cuda.set_rng_state_all(st["rng_cuda"])
    print(f"[resume] continuing {outdir} after completed epoch {st['epoch']}", flush=True)
    return st["epoch"] + 1, st["extra"]


def clear_resume(outdir: Path) -> None:
    for name in (RESUME, RESUME + ".tmp"):
        (outdir / name).unlink(missing_ok=True)
