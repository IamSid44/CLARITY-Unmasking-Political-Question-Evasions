#!/usr/bin/env python3
"""Blackwell environment verification for HiGrEC.

Prints and asserts on the facts that determine whether this project can run:
torch/CUDA versions, device capability, VRAM headroom on a SHARED card,
bitsandbytes 4-bit viability, and which attention backends are importable.

Per CLAUDE.md invariant 5 the GPU is shared, so this script reports FREE VRAM,
not just total, and treats a shortfall as a warning the caller must act on
rather than something to discover at OOM time.

Exit code 0 means the environment is usable. Exit code 1 means a hard blocker.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import platform
import sys

# Hard requirements. A failure here is a blocker, not a warning.
MIN_TORCH = (2, 7)
EXPECTED_CAPABILITY = (12, 0)  # sm_120, consumer/workstation Blackwell

blockers: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-free-vram-gb",
        type=float,
        default=0.0,
        help="Warn if no visible device has at least this much free VRAM.",
    )
    args = ap.parse_args()

    section("Platform")
    print(f"python            : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"executable        : {sys.executable}")
    print(f"os                : {platform.platform()}")

    section("Core packages")
    for name in [
        "torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes",
        "scikit-learn", "scipy", "numpy", "pandas", "krippendorff", "statsmodels",
        "pytest", "sentencepiece",
    ]:
        v = pkg_version(name)
        print(f"  {name:16s} {v if v else '-- NOT INSTALLED --'}")

    # ---- torch / CUDA -----------------------------------------------------
    section("Torch and CUDA")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment blocker
        print(f"FATAL: cannot import torch: {exc}")
        return 1

    print(f"torch.__version__          : {torch.__version__}")
    print(f"torch.version.cuda         : {torch.version.cuda}")
    print(f"torch.cuda.is_available()  : {torch.cuda.is_available()}")

    tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if tv < MIN_TORCH:
        blockers.append(
            f"torch {torch.__version__} < {'.'.join(map(str, MIN_TORCH))}; "
            "Blackwell/sm_120 needs >= 2.7 with cu128 wheels"
        )
    if not torch.cuda.is_available():
        blockers.append("CUDA is not available to torch")
        _summarize()
        return 1 if blockers else 0

    # ---- devices ----------------------------------------------------------
    section("CUDA devices (card is SHARED — see CLAUDE.md invariant 5)")
    n = torch.cuda.device_count()
    print(f"device_count: {n}")
    best_free = 0.0
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gb, total_gb = free_b / 1024**3, total_b / 1024**3
        used_gb = total_gb - free_gb
        best_free = max(best_free, free_gb)
        print(
            f"  [{i}] {name}\n"
            f"      capability : {cap[0]}.{cap[1]}"
            f"{'  (sm_120, as expected)' if cap == EXPECTED_CAPABILITY else '  (UNEXPECTED)'}\n"
            f"      VRAM       : total {total_gb:6.1f} GB | used {used_gb:6.1f} GB | free {free_gb:6.1f} GB"
        )
        if cap != EXPECTED_CAPABILITY:
            warnings.append(
                f"device {i} capability {cap} != expected {EXPECTED_CAPABILITY}; "
                "re-check the FlashAttention guidance below for this card"
            )

    if args.require_free_vram_gb and best_free < args.require_free_vram_gb:
        warnings.append(
            f"no device has {args.require_free_vram_gb:.1f} GB free "
            f"(best is {best_free:.1f} GB) — do not launch training until it does"
        )

    # ---- bitsandbytes -----------------------------------------------------
    section("bitsandbytes 4-bit viability")
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes {bnb.__version__} imported")
        try:
            lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16).cuda()
            out = lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            print(f"  4-bit Linear4bit forward OK, output shape {tuple(out.shape)}")
        except Exception as exc:
            warnings.append(f"bitsandbytes imports but 4-bit layer failed: {exc}")
            print(f"  4-bit layer FAILED: {exc}")
    except Exception as exc:
        warnings.append(f"bitsandbytes not usable: {exc}")
        print(f"NOT AVAILABLE: {exc}")
        print("  (Only needed for the QLoRA fallback path; the primary system is LoRA bf16.)")

    # ---- attention backends ----------------------------------------------
    section("Attention backends")
    print(
        "NOTE: FlashAttention-4 is NOT expected to work on sm_120 — consumer/workstation\n"
        "Blackwell lacks the TMEM subsystem that FA4 requires (only sm_100/sm_103 B200/B300\n"
        "get FA4). The targets for this project are FA2 or the FlashInfer backend.\n"
        "For vLLM, set VLLM_FLASH_ATTN_VERSION=2.\n"
    )
    for mod, note in [
        ("flash_attn", "FlashAttention (expect v2.x; v4 will not run on sm_120)"),
        ("flashinfer", "FlashInfer backend (preferred alternative on Blackwell)"),
        ("xformers", "xformers memory-efficient attention"),
    ]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", pkg_version(mod) or "unknown")
            print(f"  {mod:12s} AVAILABLE  version={ver}   # {note}")
            if mod == "flash_attn" and str(ver).split(".")[0] not in {"2", "3"}:
                warnings.append(
                    f"flash_attn version {ver} is not the FA2 line; verify it runs on sm_120"
                )
        except Exception:
            print(f"  {mod:12s} not installed   # {note}")

    # torch's own SDPA backends are always the safe fallback.
    try:
        from torch.nn.attention import SDPBackend  # noqa: F401

        print("  torch SDPA   AVAILABLE   # always-present fallback (math/mem-efficient/flash)")
    except Exception as exc:
        warnings.append(f"torch SDPA backends not importable: {exc}")

    return _summarize()


def _summarize() -> int:
    section("Summary")
    if blockers:
        print("BLOCKERS (must be fixed before proceeding):")
        for b in blockers:
            print(f"  - {b}")
    if warnings:
        print("WARNINGS (proceed with awareness):")
        for w in warnings:
            print(f"  - {w}")
    if not blockers and not warnings:
        print("Environment OK, no warnings.")
    elif not blockers:
        print("\nEnvironment USABLE (warnings above are not blockers).")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
