#!/usr/bin/env python3
"""LoRA fine-tune of a decoder LLM (default Qwen3-8B-Base) as a 9-way classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from qevasion.labels import EVASION_LABELS, N_EVASION, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import (DATA_CACHE, SPLIT_SEED, dev_reference_mask, load_qevasion, stratified_fold_index,
                             stratified_val_index)
from qevasion.paths import RUNS, SPLITS, TEST_CSV
from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2

SUFFIX = "\n\nHow does the answer respond to the sub-question?"


@dataclass
class Cfg:
    name: str = ""
    seed: int = 0
    model: str = "Qwen/Qwen3-8B-Base"
    max_len: int = 1024
    a_budget: int = 256
    epochs: int = 3
    lr: float = 1e-4
    warmup: float = 0.05
    wd: float = 0.0
    batch_size: int = 4
    grad_accum: int = 4
    eval_batch_size: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_checkpoint: bool = True
    val_frac: float = 0.1
    fold: int = -1
    n_folds: int = 5
    select: str = "best"
    dtype: str = "bf16"
    device: str = "cuda"
    limit_train: int = 0
    limit_eval: int = 0
    pilot_steps: int = 0
    stop_after_epoch: int = -1
    log_every: int = 20
    track: bool = False
    save_epoch_adapters: bool = True


def encode_row(tok, q: str, iq: str, a: str, cfg: Cfg) -> list[int]:
    """'Sub-question: q Full question: iq' (<= a_budget) + answer (right-truncated) + fixed suffix."""
    ids_a = tok("Sub-question: " + q.strip() + " Full question: " + iq.strip(), add_special_tokens=False)["input_ids"]
    ids_a = ids_a[: cfg.a_budget]
    mid = tok("\n\nAnswer: ", add_special_tokens=False)["input_ids"]
    suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    ids_b = tok(a.strip(), add_special_tokens=False)["input_ids"]
    room = cfg.max_len - len(ids_a) - len(mid) - len(suf)
    return ids_a + mid + ids_b[: max(room, 0)] + suf


def encode_df(tok, df: pd.DataFrame, cfg: Cfg) -> list[list[int]]:
    return [encode_row(tok, str(q), str(iq), str(a), cfg)
            for q, iq, a in zip(df["question"], df["interview_question"], df["interview_answer"])]


def batches_by_length(lengths: np.ndarray, bs: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle, then sort inside chunks of 50 batches so padding is small; shuffle batch order."""
    order = rng.permutation(len(lengths))
    chunk = bs * 50
    out = []
    for s in range(0, len(order), chunk):
        c = order[s: s + chunk]
        c = c[np.argsort(-lengths[c], kind="stable")]
        out += [c[i: i + bs] for i in range(0, len(c), bs)]
    rng.shuffle(out)
    return out


def collate(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids.to(device), att.to(device)


def build_model(cfg: Cfg):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[cfg.dtype]
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, num_labels=N_EVASION, dtype=dtype, device_map=None if cfg.device == "cpu" else {"": cfg.device})
    model.config.pad_token_id = tok.pad_token_id
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(cfg.device)
    return tok, model


@torch.no_grad()
def predict(model, seqs, pad_id, cfg: Cfg) -> np.ndarray:
    model.eval()
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(-lengths)
    out = np.zeros((len(seqs), N_EVASION), dtype=np.float32)
    for s in range(0, len(order), cfg.eval_batch_size):
        b = order[s: s + cfg.eval_batch_size]
        ids, att = collate([seqs[i] for i in b], pad_id, cfg.device)
        with autocast(cfg):
            logits = model(input_ids=ids, attention_mask=att).logits
        out[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    model.train()
    return out


def autocast(cfg: Cfg):
    if cfg.dtype == "fp32":
        return torch.autocast(device_type="cpu" if cfg.device == "cpu" else "cuda", enabled=False)
    return torch.autocast(device_type="cuda" if cfg.device != "cpu" else "cpu", dtype=torch.bfloat16)


def padding_self_check(model, seqs, pad_id, cfg: Cfg) -> float:
    """The head must read the last real token: an item's logits alone vs inside a padded batch."""
    model.eval()
    probe = sorted(seqs[:8], key=len)[:4]
    with torch.no_grad(), autocast(cfg):
        ids, att = collate(probe, pad_id, cfg.device)
        batched = model(input_ids=ids, attention_mask=att).logits.float()
        alone = torch.cat([model(input_ids=collate([s], pad_id, cfg.device)[0],
                                 attention_mask=collate([s], pad_id, cfg.device)[1]).logits.float() for s in probe])
    model.train()
    return float((batched - alone).abs().max() / (alone.abs().mean() + 1e-6))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(cfg: Cfg, outroot: Path) -> None:
    out = outroot / cfg.name / (f"fold{cfg.fold}" if cfg.fold >= 0 else f"seed{cfg.seed}")
    if (out / "metrics.json").exists() and not cfg.pilot_steps:
        print(f"[{cfg.name} {out.name}] already finished: {out / 'metrics.json'}")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    t0 = time.time()

    data = load_qevasion(DATA_CACHE)
    train, dev = data.train, data.dev
    test = pd.read_csv(TEST_CSV)
    y = encode_evasion(train["evasion_label"].tolist())
    if cfg.fold >= 0:
        # Cross-fitting: fold k of all 3,448 rows is held out and its probabilities are the out-of-fold
        # predictions (written as val_probs.npy). Choosing the epoch on that fold would leak, so the
        # last epoch is kept. Same partition as encoder.py --fold (stratified, SPLIT_SEED).
        if cfg.select != "last":
            raise SystemExit("--fold needs --select last (selecting on the held-out fold would leak)")
        val_idx = stratified_fold_index(y, cfg.n_folds, cfg.fold, SPLIT_SEED)
        frozen = SPLITS / "train_5fold.json"
        if cfg.n_folds == 5 and frozen.exists():
            assert np.array_equal(val_idx, np.array(json.loads(frozen.read_text())["folds"][cfg.fold])), \
                "fold differs from data/splits/train_5fold.json"
    elif cfg.val_frac > 0:
        val_idx = stratified_val_index(y, cfg.val_frac, SPLIT_SEED)
    else:
        val_idx = np.array([], dtype=int)
    ref = SPLITS / "train_internal_val_index.json"
    if cfg.fold < 0 and cfg.val_frac == 0.1 and ref.exists():
        assert np.array_equal(val_idx, np.array(json.loads(ref.read_text()))), "train slice differs from encoder runs"
    tr_idx = np.setdiff1d(np.arange(len(train)), val_idx)
    if cfg.limit_train:
        tr_idx = tr_idx[: cfg.limit_train]
    dev_mask = dev_reference_mask(dev)
    dev_clarity = encode_clarity(dev["clarity_label"].tolist())

    tok, model = build_model(cfg)
    pad = tok.pad_token_id
    X_tr = encode_df(tok, train.iloc[tr_idx], cfg)
    X_va = encode_df(tok, train.iloc[val_idx], cfg)
    X_dev, X_te = encode_df(tok, dev, cfg), encode_df(tok, test, cfg)
    if cfg.limit_eval:
        X_va, X_dev, X_te = X_va[: cfg.limit_eval], X_dev[: cfg.limit_eval], X_te[: cfg.limit_eval]
    y_tr, y_va = y[tr_idx], y[val_idx][: len(X_va)]
    len_tr = np.array([len(s) for s in X_tr])
    print(f"[data] train {len(X_tr)} (mean {len_tr.mean():.0f} tokens, max {len_tr.max()}), "
          f"slice {len(X_va)}, dev {len(X_dev)}, test {len(X_te)}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[model] {cfg.model}: trainable {n_tr / 1e6:.1f}M parameters", flush=True)
    diff = padding_self_check(model, X_dev, pad, cfg)
    print(f"[check] padded vs alone: max |logit difference| / mean |logit| = {diff:.4f} (should be << 1)", flush=True)
    if diff > 0.2:
        raise SystemExit("padding self-check failed: the head is not reading the last real token")

    steps_per_epoch = math.ceil(math.ceil(len(X_tr) / cfg.batch_size) / cfg.grad_accum)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.wd)
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup * steps_per_epoch * cfg.epochs),
                                            steps_per_epoch * cfg.epochs)

    history, best, start = [], {"val_macro_f1": -1.0}, 0
    resume = out / "resume.pt"
    if resume.exists() and not cfg.pilot_steps:
        from peft import set_peft_model_state_dict

        st = torch.load(resume, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["adapter"])
        for p in trainable:
            p.data = p.data.float()
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history, best, start = st["history"], st["best"], st["epoch"] + 1
        print(f"[resume] continuing after epoch {st['epoch']}", flush=True)

    tracker = None
    if cfg.track and not cfg.pilot_steps:
        from utils.tracking import Tracker

        tracker = Tracker(name=cfg.name, seed=f"fold{cfg.fold}" if cfg.fold >= 0 else cfg.seed, group=cfg.name,
                          job_type="fold" if cfg.fold >= 0 else "train",
                          config={**asdict(cfg), "steps_per_epoch": steps_per_epoch, "n_train": len(X_tr)},
                          state_dir=out)

    model.train()
    if cfg.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    if cfg.pilot_steps and cfg.device != "cpu":
        longest = np.argsort(-len_tr)[: cfg.batch_size]
        ids, att = collate([X_tr[j] for j in longest], pad, cfg.device)
        with autocast(cfg):
            logits = model(input_ids=ids, attention_mask=att).logits
        F.cross_entropy(logits.float(), torch.as_tensor(y_tr[longest], device=cfg.device)).backward()
        opt.zero_grad(set_to_none=True)
        print(f"[pilot] longest micro-batch ({int(att.sum())} tokens): peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB", flush=True)
    for epoch in range(start, cfg.epochs):
        rng = np.random.default_rng(cfg.seed * 1000 + epoch)
        batches = batches_by_length(len_tr, cfg.batch_size, rng)
        te0, running, nb, step, tok_seen = time.time(), 0.0, 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(batches):
            ids, att = collate([X_tr[j] for j in b], pad, cfg.device)
            with autocast(cfg):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy(logits.float(), torch.as_tensor(y_tr[b], device=cfg.device))
            (loss / cfg.grad_accum).backward()
            running, nb, tok_seen = running + loss.item(), nb + 1, tok_seen + int(att.sum())
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(batches):
                gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                el = time.time() - te0
                if tracker is not None and step % cfg.log_every == 0:
                    tracker.log({"train/loss_epoch_mean": running / nb, "train/lr": sched.get_last_lr()[0],
                                 "train/grad_norm": gnorm, "train/tokens_per_s": tok_seen / el,
                                 "train/epoch": epoch + step / steps_per_epoch},
                                step=epoch * steps_per_epoch + step)
                if step % cfg.log_every == 0 or (cfg.pilot_steps and step % 5 == 0):
                    print(f"  ep{epoch} step {step}/{steps_per_epoch} loss {running / nb:.4f} "
                          f"{tok_seen / el:.0f} tok/s  epoch ETA {el / step * (steps_per_epoch - step) / 60:.1f} min",
                          flush=True)
                if cfg.pilot_steps and step >= cfg.pilot_steps:
                    sec = el / step
                    peak = torch.cuda.max_memory_allocated() / 1024**3 if cfg.device != "cpu" else float("nan")
                    print(f"\n[pilot] {sec:.2f} s per optimizer step; {steps_per_epoch} steps per epoch "
                          f"-> {sec * steps_per_epoch / 60:.1f} min per epoch of training, "
                          f"~{sec * steps_per_epoch * cfg.epochs / 60:.0f} min for {cfg.epochs} epochs "
                          f"(+ ~3-5 min of evaluation per epoch); peak VRAM {peak:.1f} GiB", flush=True)
                    return

        tr_min = (time.time() - te0) / 60
        p_va, p_dev, p_te = (predict(model, X, pad, cfg) for X in (X_va, X_dev, X_te))
        val_f1 = macro_f1_single_label(p_va.argmax(1), y_va, EVASION_LABELS).macro_f1 if len(X_va) else float("nan")
        row = {"epoch": epoch, "train_loss": running / max(nb, 1), "val_macro_f1": val_f1,
               "train_minutes": tr_min, "minutes": (time.time() - te0) / 60}
        if len(X_dev) == len(dev):
            pred = p_dev.argmax(1)
            row.update({
                "dev_subtask2_macro_f1": score_subtask2(pred, dev_mask).macro_f1,
                "dev_subtask1_macro_f1": score_subtask1(leaf_to_official_clarity(pred), dev_clarity).macro_f1,
                "dev_inset_rate": float(dev_mask[np.arange(len(pred)), pred].mean()),
                "dev_classes_named": int(len(set(pred.tolist()))),
            })
        history.append(row)
        print(f"[{cfg.name} {out.name}] ep{epoch} loss={row['train_loss']:.4f} val={val_f1:.4f} "
              f"devS2={row.get('dev_subtask2_macro_f1', float('nan')):.4f} "
              f"devS1={row.get('dev_subtask1_macro_f1', float('nan')):.4f} ({row['minutes']:.1f}m)", flush=True)

        if tracker is not None:
            tracker.log({f"epoch/{k}": v for k, v in row.items()}
                        | {"epoch/peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if cfg.device != "cpu" else 0.0},
                        step=(epoch + 1) * steps_per_epoch)

        ep_dir = out / "epochs"
        ep_dir.mkdir(exist_ok=True)
        for nm, p in (("dev", p_dev), ("val", p_va), ("test", p_te)):
            np.save(ep_dir / f"ep{epoch}_{nm}_probs.npy", p)
        if cfg.save_epoch_adapters:
            model.save_pretrained(ep_dir / f"ep{epoch}_adapter")
        if cfg.select == "last" or val_f1 > best["val_macro_f1"]:
            best = dict(row)
            model.save_pretrained(out / "adapter_best")
        from peft import get_peft_model_state_dict

        torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "history": history, "best": best, "epoch": epoch},
                   resume.with_suffix(".tmp"))
        resume.with_suffix(".tmp").replace(resume)
        if cfg.stop_after_epoch == epoch:
            print("[test] stopping early to exercise resume", flush=True)
            return

    e = best["epoch"]
    for nm in ("dev", "val", "test"):
        np.save(out / f"{nm}_probs.npy", np.load(out / "epochs" / f"ep{e}_{nm}_probs.npy"))
    np.save(out / "val_index.npy", val_idx)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    result = {
        "config": asdict(cfg), "selected_epoch": e, "val_macro_f1": best["val_macro_f1"], "history": history,
        "dev_subtask2_macro_f1": best.get("dev_subtask2_macro_f1"),
        "dev_subtask1_macro_f1": best.get("dev_subtask1_macro_f1"),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if cfg.device != "cpu" else None,
        "wall_minutes": (time.time() - t0) / 60, "trainable_params": n_tr,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    resume.unlink(missing_ok=True)
    if tracker is not None:
        tracker.summary({"selected_epoch": e, "val_macro_f1": best["val_macro_f1"],
                         "dev_subtask2_macro_f1": result["dev_subtask2_macro_f1"],
                         "dev_subtask1_macro_f1": result["dev_subtask1_macro_f1"],
                         "peak_vram_gb": result["peak_vram_gb"] or 0.0, "wall_minutes": result["wall_minutes"]})
        tracker.finish()
    print(f"[done] {cfg.name} {out.name}: epoch {e} selected; dev S2 {result['dev_subtask2_macro_f1']}, "
          f"S1 {result['dev_subtask1_macro_f1']}; {result['wall_minutes']:.0f} min", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(Cfg()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v, required=(f == "name"))
    p.add_argument("--outroot", default=str(RUNS))
    a = vars(p.parse_args())
    outroot = Path(a.pop("outroot"))
    run(Cfg(**a), outroot)


if __name__ == "__main__":
    main()
