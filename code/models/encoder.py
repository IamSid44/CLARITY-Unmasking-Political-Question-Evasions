#!/usr/bin/env python3
"""Fine-tuned cross-encoder for SemEval-2026 Task 6 (CLARITY)."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from qevasion.labels import (
    EVASION_LABELS,
    N_CLARITY,
    N_EVASION,
    OFFICIAL_PARTITION_MAP,
    encode_clarity,
    encode_evasion,
    leaf_to_official_clarity,
)
from qevasion.loader import (SPLIT_SEED, dev_reference_mask, load_qevasion, stratified_fold_index,
                             stratified_val_index)
from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2

import pandas as pd

from utils.ckpt import atomic_torch_save, clear_resume, load_resume, save_resume
from utils.tracking import Tracker, load_env, push_async

from qevasion.loader import DATA_CACHE
from qevasion.paths import RUNS, TEST_CSV

N_TEST = 237


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    if len(df) != N_TEST:
        raise SystemExit(f"{TEST_CSV} has {len(df)} rows, expected {N_TEST}")
    return df

ANNOTATOR_IDS = ("85", "86", "89")

NON_REPLY = ("Declining to answer", "Claims ignorance", "Clarification")


def task_classes(task: str) -> list[int] | None:
    """Leaf indices a task distinguishes, in output order (None for the 2-way gate)."""
    if task == "leaf9":
        return list(range(N_EVASION))
    if task == "gate":
        return None
    if task == "nr3":
        return [EVASION_LABELS.index(c) for c in NON_REPLY]
    if task == "other6":
        return [i for i, c in enumerate(EVASION_LABELS) if c not in NON_REPLY]
    if task.startswith("pair:"):
        names = task[len("pair:"):].split(",")
        idx = []
        for n in names:
            hits = [i for i, c in enumerate(EVASION_LABELS) if c.lower().startswith(n.strip().lower())]
            if len(hits) != 1:
                raise SystemExit(f"--task {task}: {n!r} does not name exactly one label")
            idx.append(hits[0])
        if len(idx) != 2:
            raise SystemExit(f"--task {task}: a pair needs exactly two labels")
        return idx
    raise SystemExit(f"unknown --task {task!r}")


def task_targets(task: str, y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map 9-way labels to the task's classes; -1 marks rows outside the task."""
    if task == "gate":
        nr = np.isin(y, [EVASION_LABELS.index(c) for c in NON_REPLY])
        return nr.astype(np.int64), ["other", "Non-Reply"]
    cls = task_classes(task)
    lut = np.full(N_EVASION, -1, dtype=np.int64)
    lut[cls] = np.arange(len(cls))
    return lut[y], [EVASION_LABELS[i] for i in cls]


def build_inputs(df, mode: str) -> tuple[list[str], list[str]]:
    """Return (text_a, text_b) per row."""
    q = df["question"].astype(str).str.strip()
    a = df["interview_answer"].astype(str).str.strip()
    if mode == "qa":
        return q.tolist(), a.tolist()
    if mode == "full":
        iq = df["interview_question"].astype(str).str.strip()
        return ("Sub-question: " + q + " Full question: " + iq).tolist(), a.tolist()
    if mode == "qqa":
        iq = df["interview_question"].astype(str).str.strip()
        return (q + " [CONTEXT] " + iq).tolist(), a.tolist()
    raise ValueError(f"unknown input mode {mode!r}")


def encode_pair(
    tok, text_a: str, text_b: str, max_len: int, a_budget: int, head_frac: float
) -> tuple[list[int], list[int]]:
    """Build `[CLS] a [SEP] b [SEP]` with head+tail truncation applied to b."""
    ids_a = tok(text_a, add_special_tokens=False)["input_ids"][:a_budget]
    ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
    room = max_len - 3 - len(ids_a)
    if len(ids_b) > room:
        if head_frac >= 1.0:
            ids_b = ids_b[:room]
        else:
            n_head = int(round(room * head_frac))
            n_tail = room - n_head
            ids_b = ids_b[:n_head] + (ids_b[-n_tail:] if n_tail > 0 else [])
    ids = [tok.cls_token_id] + ids_a + [tok.sep_token_id] + ids_b + [tok.sep_token_id]
    types = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
    return ids, types


class PairDataset(Dataset):
    def __init__(self, tok, pairs_a, pairs_b, labels, clarity, annotators, cfg):
        self.items = [
            encode_pair(tok, a, b, cfg.max_len, cfg.a_budget, cfg.head_frac)
            for a, b in zip(pairs_a, pairs_b)
        ]
        self.labels = labels
        self.clarity = clarity
        self.annotators = annotators

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        ids, types = self.items[i]
        return {
            "input_ids": ids,
            "token_type_ids": types,
            "label": int(self.labels[i]) if self.labels is not None else -100,
            "clarity": int(self.clarity[i]) if self.clarity is not None else -100,
            "annotator": int(self.annotators[i]) if self.annotators is not None else 0,
        }


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    types = torch.zeros((len(batch), n), dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        types[i, :L] = torch.tensor(b["token_type_ids"])
        mask[i, :L] = 1
    return {
        "input_ids": ids,
        "token_type_ids": types,
        "attention_mask": mask,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "clarity": torch.tensor([b["clarity"] for b in batch], dtype=torch.long),
        "annotator": torch.tensor([b["annotator"] for b in batch], dtype=torch.long),
    }


class ClarityEncoder(nn.Module):
    """Cross-encoder trunk with a configurable output head."""

    def __init__(self, name: str, head: str, dropout: float, pooling: str, n_ann: int = 3,
                 n_out: int = N_EVASION):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        H = self.cfg.hidden_size
        self.head = head
        self.pooling = pooling
        self.pooler = nn.Linear(H, H)
        self.drop = nn.Dropout(dropout)
        self.cls9 = nn.Linear(H, n_out)
        self.cls3 = nn.Linear(H, 3) if head == "hier" else None
        self.branch = None
        if head == "hsoft":
            self.branch = nn.Linear(H, N_CLARITY)
            part = torch.as_tensor(OFFICIAL_PARTITION_MAP)
            for b in range(N_CLARITY):
                self.register_buffer(f"members_{b}", torch.nonzero(part == b).squeeze(1),
                                     persistent=False)
        self.ann_bias = nn.Parameter(torch.zeros(n_ann, N_EVASION)) if head == "annotator" else None

    def forward(self, input_ids, attention_mask, token_type_ids=None, annotator=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.cfg.model_type in {"deberta-v2", "deberta", "bert"}:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.enc(**kwargs).last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
        else:
            pooled = hidden[:, 0]
        pooled = self.drop(torch.tanh(self.pooler(pooled)))
        logits = self.cls9(pooled)
        if self.ann_bias is not None and annotator is not None:
            logits = logits + self.ann_bias[annotator]
        if self.branch is not None:
            log_level = F.log_softmax(self.branch(pooled).float(), dim=-1)
            z = logits.float()
            logp = torch.empty_like(z)
            for b in range(N_CLARITY):
                idx = getattr(self, f"members_{b}")
                logp[:, idx] = F.log_softmax(z[:, idx], dim=-1) + log_level[:, b:b + 1]
            logits = logp
        aux = self.cls3(pooled) if self.cls3 is not None else None
        return logits, aux

    def all_annotator_logits(self, pooled_logits: torch.Tensor) -> torch.Tensor:
        """(B, 9) unbiased logits -> (B, n_ann, 9) per-annotator logits."""
        return pooled_logits.unsqueeze(1) + self.ann_bias.unsqueeze(0)


def make_loss(kind: str, prior: np.ndarray, smoothing: float, device: str, gamma: float = 2.0):
    """Return `fn(logits, target) -> scalar`."""
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32, device=device)
    if kind == "ce":
        return lambda z, y: F.cross_entropy(z, y, label_smoothing=smoothing)
    if kind == "bal":
        return lambda z, y: F.cross_entropy(z + log_prior, y, label_smoothing=smoothing)
    if kind == "invfreq":
        w = torch.tensor(1.0 / (prior + 1e-12), dtype=torch.float32, device=device)
        w = w / w.mean()
        return lambda z, y: F.cross_entropy(z, y, weight=w, label_smoothing=smoothing)
    if kind in ("focal", "bal_focal"):
        shift = log_prior if kind == "bal_focal" else torch.zeros_like(log_prior)

        def focal(z, y):
            logp = F.log_softmax(z + shift, dim=-1).gather(1, y[:, None]).squeeze(1)
            return (-((1.0 - logp.exp()) ** gamma) * logp).mean()

        return focal
    raise ValueError(f"unknown loss {kind!r}")


def param_groups(model: ClarityEncoder, lr: float, head_lr: float, decay: float, wd: float):
    """Layer-wise learning-rate decay."""
    n_layers = model.cfg.num_hidden_layers
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")

    def wd_of(name: str) -> float:
        return 0.0 if any(k in name for k in no_decay) else wd

    def depth_of(name: str) -> int:
        if not name.startswith("enc."):
            return n_layers + 1
        if "embeddings" in name:
            return 0
        for part in name.split("."):
            if part.isdigit():
                return int(part) + 1
        return n_layers

    groups: dict[tuple[int, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups.setdefault((depth_of(name), wd_of(name)), []).append(p)
    out = []
    for (depth, w), params in groups.items():
        base = head_lr if depth == n_layers + 1 else lr * (decay ** (n_layers + 1 - depth))
        out.append({"params": params, "lr": base, "weight_decay": w})
    return out


def reinit_top_layers(model: ClarityEncoder, k: int) -> None:
    """Re-initialise the top `k` transformer layers."""
    if k <= 0:
        return
    layers = model.enc.encoder.layer
    for layer in layers[-k:]:
        layer.apply(model.enc._init_weights)


@torch.no_grad()
def predict(model, loader, device, n_ann: int) -> np.ndarray:
    """Return (N, 9) probabilities, or (N, n_ann, 9) for the annotator head."""
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], annotator=None
            )
        logits = logits.float()
        if model.ann_bias is not None:
            probs = torch.softmax(model.all_annotator_logits(logits), dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
        out.append(probs.cpu().numpy())
    if not out:
        return np.zeros((0, model.cls9.out_features), dtype=np.float32)
    return np.concatenate(out, axis=0)


def inset_diagnostics(y_pred: np.ndarray, gold_mask: np.ndarray) -> dict:
    """In-set rate and class coverage -- the two terms macro-F1 decomposes into."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    known = y_pred >= 0
    hit = np.zeros(len(y_pred), dtype=bool)
    hit[known] = gold_mask[np.arange(len(y_pred))[known], y_pred[known]]
    named = sorted(set(y_pred[known].tolist()))
    return {
        "inset_rate": float(hit.mean()),
        "classes_named": len(named),
        "coverage_ceiling": len(named) / gold_mask.shape[1],
        "never_named": [EVASION_LABELS[c] for c in range(gold_mask.shape[1]) if c not in named],
    }


def consensus(probs: np.ndarray) -> np.ndarray:
    """Collapse per-annotator probabilities to one posterior by averaging."""
    return probs.mean(axis=1) if probs.ndim == 3 else probs


@dataclass
class Cfg:
    name: str
    model: str = "microsoft/deberta-v3-large"
    input: str = "qa"
    max_len: int = 512
    a_budget: int = 96
    head_frac: float = 1.0
    loss: str = "ce"
    smoothing: float = 0.0
    focal_gamma: float = 2.0
    head: str = "flat"
    aux_weight: float = 0.3
    pooling: str = "cls"
    dropout: float = 0.1
    lr: float = 1e-5
    head_lr: float = 1e-4
    llrd: float = 0.95
    wd: float = 0.01
    epochs: int = 8
    batch_size: int = 16
    grad_accum: int = 1
    warmup: float = 0.1
    reinit_top: int = 0
    seed: int = 0
    val_frac: float = 0.1
    grad_checkpoint: bool = True
    max_vram_gb: float = 24.0
    save_model: bool = False
    select: str = "best"
    fold: int = -1
    n_folds: int = 5
    max_train: int = 0
    push_to_hub: bool = False
    task: str = "leaf9"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_vram(max_gb: float) -> None:
    """The card is shared with other users' jobs: fail fast, never OOM a neighbour."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1024**3
    if free_gb < max_gb:
        raise SystemExit(
            f"FAIL FAST: {free_gb:.1f} GB free, need {max_gb:.1f} GB. "
            "Not starting rather than OOM-ing a shared card."
        )
    print(f"[vram] {free_gb:.1f} GB free of {total_b / 1024**3:.1f} GB")


def run(cfg: Cfg, outdir: Path) -> dict:
    device = "cuda"
    tag = f"f{cfg.fold}" if cfg.fold >= 0 else f"s{cfg.seed}"
    check_vram(cfg.max_vram_gb)
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    splits = load_qevasion(DATA_CACHE)
    train_df, dev_df = splits.train, splits.dev

    y_train = encode_evasion(train_df["label" if "label" in train_df else "evasion_label"].tolist())
    if (y_train < 0).any():
        raise SystemExit("unrecognised training label -- vocabulary drift, refusing to train")
    c_train = leaf_to_official_clarity(y_train)
    ann_train = np.asarray(
        [ANNOTATOR_IDS.index(str(a)) for a in train_df["annotator_id"].astype(str)], dtype=np.int64
    )

    y_task, task_names = task_targets(cfg.task, y_train)
    n_out = len(task_names)
    leaf9 = cfg.task == "leaf9"
    if not leaf9 and (cfg.head != "flat" or cfg.fold >= 0):
        raise SystemExit("--task other than leaf9 needs --head flat and no --fold")

    if cfg.fold >= 0:
        val_idx = stratified_fold_index(y_train, cfg.n_folds, cfg.fold, SPLIT_SEED)
    elif cfg.val_frac <= 0:
        if cfg.select != "last":
            raise SystemExit("--val-frac 0 needs --select last")
        val_idx = np.zeros(0, dtype=np.int64)
    else:
        val_idx = stratified_val_index(y_train, cfg.val_frac, SPLIT_SEED)
    tr_mask = np.ones(len(y_train), dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]
    tr_idx = tr_idx[y_task[tr_idx] >= 0]
    val_idx = val_idx[y_task[val_idx] >= 0]
    if cfg.max_train > 0:
        tr_idx = np.random.default_rng(cfg.seed).permutation(tr_idx)[: cfg.max_train]

    test_df = load_test()
    tracker = Tracker(
        name=cfg.name,
        seed=cfg.seed if cfg.fold < 0 else f"fold{cfg.fold}",
        group=cfg.name,
        job_type="fold" if cfg.fold >= 0 else "train",
        config=asdict(cfg),
        state_dir=outdir,
    )

    tok = AutoTokenizer.from_pretrained(cfg.model)
    a_tr, b_tr = build_inputs(train_df, cfg.input)
    a_dev, b_dev = build_inputs(dev_df, cfg.input)
    a_te, b_te = build_inputs(test_df, cfg.input)

    ds_tr = PairDataset(
        tok,
        [a_tr[i] for i in tr_idx],
        [b_tr[i] for i in tr_idx],
        y_task[tr_idx],
        c_train[tr_idx],
        ann_train[tr_idx],
        cfg,
    )
    ds_va = PairDataset(
        tok,
        [a_tr[i] for i in val_idx],
        [b_tr[i] for i in val_idx],
        y_task[val_idx],
        c_train[val_idx],
        ann_train[val_idx],
        cfg,
    )
    ds_dev = PairDataset(tok, a_dev, b_dev, None, None, None, cfg)
    ds_te = PairDataset(tok, a_te, b_te, None, None, None, cfg)

    pad = tok.pad_token_id
    g = torch.Generator().manual_seed(cfg.seed)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate(b, pad),
        drop_last=False,
    )
    dl_va = DataLoader(ds_va, batch_size=32, collate_fn=lambda b: collate(b, pad))
    dl_dev = DataLoader(ds_dev, batch_size=32, collate_fn=lambda b: collate(b, pad))
    dl_te = DataLoader(ds_te, batch_size=32, collate_fn=lambda b: collate(b, pad))

    model = ClarityEncoder(cfg.model, cfg.head, cfg.dropout, cfg.pooling, n_out=n_out).to(device)
    reinit_top_layers(model, cfg.reinit_top)
    if cfg.grad_checkpoint:
        model.enc.gradient_checkpointing_enable()

    prior = np.bincount(y_task[tr_idx], minlength=n_out).astype(np.float64)
    prior = prior / prior.sum()
    loss_fn = make_loss(cfg.loss, prior, cfg.smoothing, device, cfg.focal_gamma)

    steps_per_epoch = math.ceil(len(dl_tr) / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(
        param_groups(model, cfg.lr, cfg.head_lr, cfg.llrd, cfg.wd), eps=1e-6, betas=(0.9, 0.98)
    )
    sched = get_cosine_schedule_with_warmup(opt, int(total * cfg.warmup), total)

    dev_mask = dev_reference_mask(dev_df)
    dev_clarity = encode_clarity(dev_df["clarity_label"].tolist())

    history, best = [], {"val_macro_f1": -1.0}
    best_dev_probs = best_val_probs = best_test_probs = None
    t0 = time.time()

    start_epoch, extra = load_resume(outdir, model=model, opt=opt, sched=sched, gen=g)
    if extra is not None:
        history, best = extra["history"], extra["best"]
        best_dev_probs, best_val_probs, best_test_probs = extra["best_probs"]
        t0 -= extra["elapsed_s"]

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    annotator=batch["annotator"] if cfg.head == "annotator" else None,
                )
                loss = loss_fn(logits.float(), batch["label"])
                if aux is not None:
                    loss = loss + cfg.aux_weight * F.cross_entropy(aux.float(), batch["clarity"])
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            running += float(loss)
            nb += 1

        val_probs = predict(model, dl_va, device, len(ANNOTATOR_IDS))
        dev_probs = predict(model, dl_dev, device, len(ANNOTATOR_IDS))
        test_probs = predict(model, dl_te, device, len(ANNOTATOR_IDS))

        if len(val_idx):
            val_pred = consensus(val_probs).argmax(1)
            val_score = macro_f1_single_label(val_pred, y_task[val_idx], tuple(task_names)).macro_f1
        else:
            val_score = float("nan")
        row = {
            "epoch": epoch,
            "train_loss": running / max(nb, 1),
            "val_macro_f1": val_score,
            "minutes": (time.time() - t0) / 60,
        }
        if leaf9:
            dev_pred = consensus(dev_probs).argmax(1)
            s2 = score_subtask2(dev_pred, dev_mask)
            s1 = score_subtask1(leaf_to_official_clarity(dev_pred), dev_clarity)
            diag = inset_diagnostics(dev_pred, dev_mask)
            row.update({
                "dev_subtask2_macro_f1": s2.macro_f1,
                "dev_subtask1_macro_f1": s1.macro_f1,
                "dev_inset_rate": diag["inset_rate"],
                "dev_classes_named": diag["classes_named"],
            })
            dev_txt = (f"devS2={s2.macro_f1:.4f} devS1={s1.macro_f1:.4f} "
                       f"inset={diag['inset_rate']:.3f} named={diag['classes_named']}/9 ")
        else:
            dev_txt = f"task={cfg.task} "
        history.append(row)
        print(
            f"[{cfg.name} {tag}] ep{epoch} loss={row['train_loss']:.4f} "
            f"val={val_score:.4f} {dev_txt}({row['minutes']:.1f}m)",
            flush=True,
        )
        tracker.log(row, step=epoch)
        if cfg.select == "last" or val_score > best["val_macro_f1"]:
            best = dict(row)
            best_dev_probs, best_val_probs, best_test_probs = dev_probs, val_probs, test_probs
            if cfg.save_model:
                atomic_torch_save(
                    {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                     for k, v in model.state_dict().items()},
                    outdir / "model.pt",
                )
        save_resume(outdir, epoch=epoch, model=model, opt=opt, sched=sched, gen=g, extra={
            "history": history, "best": best,
            "best_probs": (best_dev_probs, best_val_probs, best_test_probs),
            "elapsed_s": time.time() - t0,
        })

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "dev_probs.npy", best_dev_probs)
    np.save(outdir / "val_probs.npy", best_val_probs)
    np.save(outdir / "val_index.npy", val_idx)
    np.save(outdir / "test_probs.npy", best_test_probs)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    result = {
        "config": asdict(cfg),
        "task_classes": task_names,
        "selected_epoch": best["epoch"],
        "val_macro_f1": best["val_macro_f1"],
        "history": history,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "wall_minutes": (time.time() - t0) / 60,
    }
    if leaf9:
        dev_pred = consensus(best_dev_probs).argmax(1)
        s2 = score_subtask2(dev_pred, dev_mask)
        s1 = score_subtask1(leaf_to_official_clarity(dev_pred), dev_clarity)
        result.update({
            "dev_subtask2_macro_f1": s2.macro_f1,
            "dev_subtask1_macro_f1": s1.macro_f1,
            "dev_inset": inset_diagnostics(dev_pred, dev_mask),
            "dev_subtask2_per_class": {
                k: {"f1": v.f1, "support": v.support, "tp": v.tp, "fp": v.fp, "fn": v.fn}
                for k, v in s2.per_class.items()
            },
        })
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
    clear_resume(outdir)
    summary = {
        "selected_epoch": result["selected_epoch"],
        "best_val_macro_f1": result["val_macro_f1"],
        "peak_vram_gb": result["peak_vram_gb"],
    }
    if leaf9:
        summary.update({
            "dev_subtask2_macro_f1": result["dev_subtask2_macro_f1"],
            "dev_subtask1_macro_f1": result["dev_subtask1_macro_f1"],
            "dev_inset_rate": result["dev_inset"]["inset_rate"],
        })
    tracker.summary(summary)
    tracker.finish()
    if cfg.push_to_hub:
        push_async(outdir)
    dev_txt = (f"devS2={result['dev_subtask2_macro_f1']:.4f} devS1={result['dev_subtask1_macro_f1']:.4f} "
               if leaf9 else f"task={cfg.task} ")
    print(f"[{cfg.name} {tag}] DONE  {dev_txt}"
          f"peak={result['peak_vram_gb']:.1f}GB  {result['wall_minutes']:.1f}m", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for f, v in asdict(Cfg(name="x")).items():
        if f == "name":
            p.add_argument("--name", required=True)
        elif isinstance(v, bool):
            p.add_argument(f"--{f.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--outroot", default=str(RUNS))
    args = p.parse_args()
    cfg = Cfg(**{k: v for k, v in vars(args).items() if k != "outroot"})
    load_env()
    sub = f"fold{cfg.fold}" if cfg.fold >= 0 else f"seed{cfg.seed}"
    outdir = Path(args.outroot) / cfg.name / sub
    outdir.mkdir(parents=True, exist_ok=True)
    run(cfg, outdir)


if __name__ == "__main__":
    main()
