#!/usr/bin/env python3
"""Stage 2: a cross-encoder that re-ranks Stage 1's top-K candidate labels.

RESULT: NEGATIVE. With the same backbone and budget as Stage 1 (DeBERTa-v3-large,
8 epochs, 5 seeds) it scores 0.329 +/- 0.039 dev Subtask-2 macro-F1 against 0.365
for the Stage-1 ensemble it re-ranks, and it also loses on the held-out train
slice (0.300 vs 0.405). Kept as a documented ablation and as the slot an LLM
chooser will occupy later. Write-up: reports/03_reranker_ablation.md.

Why
---
The Stage-1 baseline's *ranking* is much better than its *top-1 decision*. On
dev (seed 0), the reference set appears in its top-1 for 52.9% of items but in
its top-3 for 88.3%, and a perfect chooser among those three would score 0.729
macro-F1 (reports/02_experiment_log.md). This model is that chooser.

How it handles every item having a different top-K
--------------------------------------------------
The candidate class is an INPUT, not an output slot. The model is a scoring
function

    s(sub-question, answer, candidate) -> one real number

where `candidate` enters as text: the class name plus a fixed one-sentence
description (LABEL_DESCRIPTIONS below -- a lookup table, written once, never
generated). For each item we call it once per candidate in that item's own top-K
and take the argmax. Because nothing in the network is tied to a class index,
item 17 can have {Dodging, General, Deflection} and item 18 {Explicit, Implicit,
Partial/half-answer} with no special handling. This is the standard retrieval
re-ranker set-up (monoBERT; Nogueira & Cho, 2019), with labels in place of
documents.

Training signal: for each training item, a softmax over its K candidate scores,
with the gold candidate as the target.

Cross-fitting is not optional
-----------------------------
Training candidates MUST come from out-of-fold Stage-1 predictions
(`encoder.py --fold k`). The baseline drives its training loss to ~0.6, i.e. it
has memorised its own training rows; in-sample, the gold label would almost
always be ranked first, and a re-ranker trained on that learns "always pick
candidate 1" and is useless at test time. Candidates for dev and test come from
the Stage-1 seed ensemble, which never saw those rows.

When the gold label is not in a training item's top-K, it replaces the K-th
candidate, so every training item has a positive. At inference nothing is
injected: an item whose reference set is not in its top-K cannot be recovered,
which is exactly the gap between the 0.883 ceiling and 1.0.

Outputs (runs/<name>/seed<k>/)
------------------------------
  {val,dev,test}_cands.npy   (N, K) candidate class indices
  {val,dev,test}_scores.npy  (N, K) re-ranker scores at the selected epoch
  metrics.json               per-epoch history, dev scores for both decision
                             variants (pure re-ranker; re-ranker + Stage-1 prior)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from encoder import (
    DATA_CACHE,
    SPLIT_SEED,
    build_inputs,
    check_vram,
    consensus,
    encode_pair,
    inset_diagnostics,
    load_test,
    param_groups,
    set_seed,
    stratified_val_index,
)
from qevasion.labels import (
    EVASION_LABELS,
    N_EVASION,
    encode_clarity,
    encode_evasion,
    leaf_to_official_clarity,
)
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2
from ckpt import atomic_torch_save, clear_resume, load_resume, save_resume
from tracking import Tracker, load_env, push_async

RUNS = Path(os.environ.get("CLARITY_RUNS", Path(__file__).resolve().parent / "runs"))

# Paraphrased from the organizers' taxonomy table (Thomas et al., EMNLP 2024,
# Materials/.../EMNLP_2024_Baseline.pdf). Written for this project; no competitor
# prompt text. Kept short on purpose: it shares a
# 512-token window with the sub-question and the answer.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "Explicit": "the requested information is stated directly, in the form asked for",
    "Implicit": "the requested information is given, but only implied rather than stated outright",
    "Dodging": "the question is ignored altogether and the answer goes to another topic",
    "General": "the answer stays on topic but is too vague or general to give the specific information asked for",
    "Deflection": "the answer starts on topic, then shifts focus and makes a different point than the one asked about",
    "Partial/half-answer": "only one component of the requested information is given; the rest is left out",
    "Declining to answer": "the question is acknowledged but the speaker refuses, directly or indirectly, to answer it",
    "Claims ignorance": "the speaker says they do not know the answer",
    "Clarification": "instead of answering, the speaker asks for the question to be clarified or repeated",
}
assert set(LABEL_DESCRIPTIONS) == set(EVASION_LABELS)


def candidate_text(c: int, question: str) -> str:
    name = EVASION_LABELS[c]
    return f"Candidate: {name} -- {LABEL_DESCRIPTIONS[name]}. Question: {question}"


# --------------------------------------------------------------------------
# Stage-1 candidates
# --------------------------------------------------------------------------


def load_stage1(n_train: int, oof_name: str, base_name: str):
    """Out-of-fold train probabilities and seed-ensembled dev/test probabilities."""
    oof = np.full((n_train, N_EVASION), np.nan)
    folds = sorted((RUNS / oof_name).glob("fold*"))
    for f in folds:
        idx = np.load(f / "val_index.npy")
        oof[idx] = consensus(np.load(f / "val_probs.npy"))
    if not folds or np.isnan(oof).any():
        raise SystemExit(
            f"incomplete out-of-fold predictions under runs/{oof_name}: "
            f"{len(folds)} folds, {int(np.isnan(oof).any(1).sum())} rows missing"
        )
    seeds = sorted((RUNS / base_name).glob("seed*/test_probs.npy"))
    if not seeds:
        raise SystemExit(f"no Stage-1 test_probs.npy under runs/{base_name}")
    dev = np.mean([consensus(np.load(s.parent / "dev_probs.npy")) for s in seeds], axis=0)
    test = np.mean([consensus(np.load(s)) for s in seeds], axis=0)
    print(f"[stage1] {len(folds)} OOF folds, {len(seeds)}-seed ensemble for dev/test")
    return oof, dev, test


def top_k(p: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-p, axis=1)[:, :k]


# --------------------------------------------------------------------------
# Data / model
# --------------------------------------------------------------------------


class RerankDataset(Dataset):
    """One item = K (candidate, sub-question, answer) sequences + gold position."""

    def __init__(self, tok, questions, answers, cands, gold, cfg):
        self.items = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            seqs = [
                encode_pair(tok, candidate_text(int(c), q), a, cfg.max_len, cfg.a_budget, cfg.head_frac)
                for c in cands[i]
            ]
            self.items.append((seqs, int(gold[i]) if gold is not None else -100))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    seqs = [s for item, _ in batch for s in item]
    n = max(len(ids) for ids, _ in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    types = torch.zeros((len(seqs), n), dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for j, (x, t) in enumerate(seqs):
        ids[j, : len(x)] = torch.tensor(x)
        types[j, : len(t)] = torch.tensor(t)
        mask[j, : len(x)] = 1
    gold = torch.tensor([g for _, g in batch], dtype=torch.long)
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": mask, "gold": gold}


class RerankModel(nn.Module):
    def __init__(self, name: str, dropout: float):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(name)
        # fp32 load is load-bearing: see encoder.ClarityEncoder and README.md ("A bug that silently prevented training").
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        H = self.cfg.hidden_size
        self.pooler = nn.Linear(H, H)
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.cfg.model_type in {"deberta-v2", "deberta", "bert"}:
            kwargs["token_type_ids"] = token_type_ids
        h = self.enc(**kwargs).last_hidden_state[:, 0]
        return self.score(self.drop(torch.tanh(self.pooler(h)))).squeeze(-1)


@torch.no_grad()
def score_all(model, loader, device, k: int) -> np.ndarray:
    model.eval()
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                      b["token_type_ids"].to(device))
        out.append(s.float().view(-1, k).cpu().numpy())
    return np.concatenate(out)


def choose(cands: np.ndarray, scores: np.ndarray, prior: np.ndarray | None, beta: float) -> np.ndarray:
    """argmax over each item's candidates; optionally add beta * log p_stage1."""
    s = scores.copy()
    if prior is not None and beta:
        s = s + beta * np.log(np.take_along_axis(prior, cands, axis=1) + 1e-12)
    return np.take_along_axis(cands, s.argmax(1)[:, None], axis=1)[:, 0]


# --------------------------------------------------------------------------


@dataclass
class RCfg:
    name: str = "R1_rerank"
    model: str = "microsoft/deberta-v3-large"
    stage1_oof: str = "F0_oof"
    stage1_base: str = "L0_large_base"
    k: int = 3
    max_len: int = 512
    a_budget: int = 128
    head_frac: float = 1.0
    dropout: float = 0.1
    lr: float = 1e-5
    head_lr: float = 1e-4
    llrd: float = 0.95
    wd: float = 0.01
    # 8, matching Stage 1. A 3-epoch run (2026-09-20) was UNDERTRAINED: train loss
    # moved only 1.044 -> 0.967 against ln(3)=1.099 for random guessing, and the
    # result was worse than Stage 1 alone (0.249 vs 0.365 dev macro-F1).
    epochs: int = 8
    items_per_batch: int = 4
    grad_accum: int = 4
    warmup: float = 0.1
    seed: int = 0
    val_frac: float = 0.1
    max_vram_gb: float = 16.0
    max_train: int = 0
    save_model: bool = True
    push_to_hub: bool = False


# The weight on the Stage-1 prior. The grid must reach far enough that the fusion
# can fall back to Stage 1's own ordering: at large beta the prior term dominates
# the (bounded) re-ranker score, so `rerank + prior` degenerates to Stage-1 argmax
# and the combination can never score below Stage 1. The first attempt stopped at
# 2.0 and every seed chose that edge value, i.e. the fit wanted more prior than it
# was allowed, and the result came out below Stage 1.
BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def run(cfg: RCfg, outdir: Path) -> dict:
    device = "cuda"
    check_vram(cfg.max_vram_gb)
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    sp = load_qevasion(DATA_CACHE)
    train_df, dev_df, test_df = sp.train, sp.dev, load_test()
    y = encode_evasion(train_df["evasion_label"].tolist())
    oof, p_dev, p_test = load_stage1(len(y), cfg.stage1_oof, cfg.stage1_base)

    val_idx = stratified_val_index(y, cfg.val_frac, SPLIT_SEED)
    mask = np.ones(len(y), bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    if cfg.max_train > 0:
        tr_idx = np.random.default_rng(cfg.seed).permutation(tr_idx)[: cfg.max_train]

    c_all = top_k(oof, cfg.k)
    # Training candidates: inject gold into the K-th slot when Stage 1 missed it.
    c_tr = c_all[tr_idx].copy()
    miss = ~(c_tr == y[tr_idx, None]).any(1)
    c_tr[miss, -1] = y[tr_idx][miss]
    g_tr = (c_tr == y[tr_idx, None]).argmax(1)
    c_va = c_all[val_idx]
    c_dev, c_te = top_k(p_dev, cfg.k), top_k(p_test, cfg.k)
    print(f"[cands] train: gold outside top-{cfg.k} for {miss.mean():.1%} (injected); "
          f"internal-val hit@{cfg.k} = {(c_va == y[val_idx, None]).any(1).mean():.3f}")

    q_tr, a_tr = build_inputs(train_df, "qa")
    q_dev, a_dev = build_inputs(dev_df, "qa")
    q_te, a_te = build_inputs(test_df, "qa")

    tok = AutoTokenizer.from_pretrained(cfg.model)
    pad = tok.pad_token_id
    ds_tr = RerankDataset(tok, [q_tr[i] for i in tr_idx], [a_tr[i] for i in tr_idx], c_tr, g_tr, cfg)
    ds_va = RerankDataset(tok, [q_tr[i] for i in val_idx], [a_tr[i] for i in val_idx], c_va, None, cfg)
    ds_dev = RerankDataset(tok, q_dev, a_dev, c_dev, None, cfg)
    ds_te = RerankDataset(tok, q_te, a_te, c_te, None, cfg)
    g = torch.Generator().manual_seed(cfg.seed)
    col = lambda b: collate(b, pad)  # noqa: E731
    dl_tr = DataLoader(ds_tr, batch_size=cfg.items_per_batch, shuffle=True, generator=g, collate_fn=col)
    dl_va = DataLoader(ds_va, batch_size=8, collate_fn=col)
    dl_dev = DataLoader(ds_dev, batch_size=8, collate_fn=col)
    dl_te = DataLoader(ds_te, batch_size=8, collate_fn=col)

    model = RerankModel(cfg.model, cfg.dropout).to(device)
    model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.head_lr, cfg.llrd, cfg.wd),
                            eps=1e-6, betas=(0.9, 0.98))
    total = math.ceil(len(dl_tr) / cfg.grad_accum) * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * cfg.warmup), total)

    dev_mask = dev_reference_mask(dev_df)
    dev_clar = encode_clarity(dev_df["clarity_label"].tolist())
    tracker = Tracker(name=cfg.name, seed=cfg.seed, group=cfg.name, job_type="rerank",
                      config=asdict(cfg), state_dir=outdir)

    history, best, saved = [], {"val_macro_f1": -1.0}, None
    t0 = time.time()
    start_epoch, extra = load_resume(outdir, model=model, opt=opt, sched=sched, gen=g)
    if extra is not None:
        history, best, saved = extra["history"], extra["best"], extra["saved"]
        t0 -= extra["elapsed_s"]
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(dl_tr):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["token_type_ids"].to(device))
            loss = F.cross_entropy(s.float().view(-1, cfg.k), b["gold"].to(device))
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss.detach())
            nb += 1

        s_va, s_dev, s_te = (score_all(model, d, device, cfg.k) for d in (dl_va, dl_dev, dl_te))
        pred_va = choose(c_va, s_va, None, 0.0)
        val_f1 = macro_f1_single_label(pred_va, y[val_idx], EVASION_LABELS).macro_f1
        pred_dev = choose(c_dev, s_dev, None, 0.0)
        d2 = score_subtask2(pred_dev, dev_mask).macro_f1
        d1 = score_subtask1(leaf_to_official_clarity(pred_dev), dev_clar).macro_f1
        diag = inset_diagnostics(pred_dev, dev_mask)
        row = {"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_macro_f1": val_f1,
               "dev_subtask2_macro_f1": d2, "dev_subtask1_macro_f1": d1,
               "dev_inset_rate": diag["inset_rate"], "dev_classes_named": diag["classes_named"],
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        tracker.log(row, step=epoch)
        print(f"[{cfg.name} s{cfg.seed}] ep{epoch} loss={row['train_loss']:.4f} val={val_f1:.4f} "
              f"devS2={d2:.4f} devS1={d1:.4f} inset={diag['inset_rate']:.3f} "
              f"named={diag['classes_named']}/9 ({row['minutes']:.1f}m)", flush=True)
        if val_f1 > best["val_macro_f1"]:
            best, saved = dict(row), (s_va, s_dev, s_te)
            if cfg.save_model:
                atomic_torch_save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                                   for k, v in model.state_dict().items()}, outdir / "model.pt")
        save_resume(outdir, epoch=epoch, model=model, opt=opt, sched=sched, gen=g,
                    extra={"history": history, "best": best, "saved": saved,
                           "elapsed_s": time.time() - t0})

    s_va, s_dev, s_te = saved
    # Fusion weight for the Stage-1 prior, chosen on the INTERNAL val split only.
    beta_scores = {b: macro_f1_single_label(choose(c_va, s_va, oof[val_idx], b), y[val_idx],
                                            EVASION_LABELS).macro_f1 for b in BETA_GRID}
    beta = max(beta_scores, key=beta_scores.get)

    report = {}
    for tag, b in (("rerank", 0.0), ("rerank_plus_prior", beta)):
        pd_ = choose(c_dev, s_dev, p_dev, b)
        report[tag] = {
            "beta": b,
            "dev_subtask2_macro_f1": score_subtask2(pd_, dev_mask).macro_f1,
            "dev_subtask1_macro_f1": score_subtask1(leaf_to_official_clarity(pd_), dev_clar).macro_f1,
            "dev_inset": inset_diagnostics(pd_, dev_mask),
        }
    stage1_pred = p_dev.argmax(1)
    report["stage1_ensemble_argmax"] = {
        "dev_subtask2_macro_f1": score_subtask2(stage1_pred, dev_mask).macro_f1,
        "dev_subtask1_macro_f1": score_subtask1(leaf_to_official_clarity(stage1_pred), dev_clar).macro_f1,
        "dev_inset": inset_diagnostics(stage1_pred, dev_mask),
    }
    report["dev_hit_at_k"] = float(np.mean([dev_mask[i, c_dev[i]].any() for i in range(len(c_dev))]))
    # Did re-ranking help at all, judged on the internal split only? Stage-1's own
    # ordering is candidate 0, so "always candidate 0" is Stage 1 restricted to the
    # shortlist; the re-ranker has to beat that to have earned its place.
    stage1_val = macro_f1_single_label(c_va[:, 0], y[val_idx], EVASION_LABELS).macro_f1
    report["internal_val"] = {"stage1_top1": stage1_val, "rerank": best["val_macro_f1"],
                              "beta_grid": beta_scores, "selected_beta": beta,
                              "rerank_beats_stage1": bool(best["val_macro_f1"] > stage1_val)}

    for split, c, s in (("val", c_va, s_va), ("dev", c_dev, s_dev), ("test", c_te, s_te)):
        np.save(outdir / f"{split}_cands.npy", c)
        np.save(outdir / f"{split}_scores.npy", s)
    result = {"config": asdict(cfg), "selected_epoch": best["epoch"], "val_macro_f1": best["val_macro_f1"],
              "beta_grid_internal_val": beta_scores, "selected_beta": beta, "dev": report,
              "history": history, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
              "wall_minutes": (time.time() - t0) / 60}
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
    clear_resume(outdir)
    tracker.summary({"selected_epoch": best["epoch"], "selected_beta": beta,
                     "dev_s2_rerank": report["rerank"]["dev_subtask2_macro_f1"],
                     "dev_s2_rerank_plus_prior": report["rerank_plus_prior"]["dev_subtask2_macro_f1"],
                     "dev_s2_stage1": report["stage1_ensemble_argmax"]["dev_subtask2_macro_f1"],
                     "dev_hit_at_k": report["dev_hit_at_k"]})
    tracker.finish()
    print(f"[{cfg.name} s{cfg.seed}] DONE  stage1 devS2={report['stage1_ensemble_argmax']['dev_subtask2_macro_f1']:.4f} "
          f"| rerank devS2={report['rerank']['dev_subtask2_macro_f1']:.4f} "
          f"| +prior(b={beta}) devS2={report['rerank_plus_prior']['dev_subtask2_macro_f1']:.4f} "
          f"| hit@{cfg.k}={report['dev_hit_at_k']:.3f}  {result['wall_minutes']:.1f}m", flush=True)
    if cfg.push_to_hub:
        push_async(outdir)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(RCfg()).items():
        flag = f"--{f.replace('_', '-')}"
        if isinstance(v, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(flag, type=type(v), default=v)
    args = p.parse_args()
    cfg = RCfg(**vars(args))
    load_env()
    outdir = RUNS / cfg.name / f"seed{cfg.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    run(cfg, outdir)


if __name__ == "__main__":
    main()
