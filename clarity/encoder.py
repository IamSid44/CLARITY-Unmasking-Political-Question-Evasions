#!/usr/bin/env python3
"""Fine-tuned cross-encoder for SemEval-2026 Task 6 (CLARITY).

This is the *end-to-end fine-tuned* track. It was built alongside the team's
earlier frozen-backbone `higrec` track (now archived), reusing that track's data
layer, label vocabulary and scorer replica (carried over as `qevasion/`), and
supplying the one thing that track did not have: a backbone actually adapted to
the task.

Why this exists
---------------
The higrec design record (`reasoning.md` D-02, archived) freezes the trunk and trains only output heads. That
is a sound way to compare heads cheaply, and D-02 says so honestly: it also
records that "the sample-efficiency argument is about end-to-end learning" and
that a confirmatory fine-tune is "not optional if the headline claim is to
stand." Nothing downstream -- head comparisons, decision rules, calibration --
means much until the posterior it operates on comes from a trained model. This
file produces that posterior.

Design
------
Everything below is an axis the shared-task overview report identifies as
load-bearing, exposed as a flag so an experiment can move one
at a time:

  input        which fields enter the cross-encoder
  truncation   head-only vs head+tail (Sun et al., CCL 2019)
  loss         CE / balanced softmax (Ren et al., NeurIPS 2020) / inverse-freq
  head         flat 9-way / +auxiliary clarity / per-annotator bias
               (Davani et al., TACL 2022)

The decision rule is *not* an axis here. Every run saves its per-item
probability matrices; choosing what to do with them is post-hoc and free, and
lives in `decide.py`.

Outputs per run (no model weights unless --save-model):
  config.json      exact configuration, for provenance
  metrics.json     per-epoch internal-val + dev scores, and the selected epoch
  dev_probs.npy    (308, 9) or (308, n_annotators, 9) for the annotator head
  val_probs.npy    internal-validation probabilities, same convention
  val_index.npy    row indices of the internal-validation split within train
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2

import pandas as pd

from ckpt import atomic_torch_save, clear_resume, load_resume, save_resume
from tracking import Tracker, load_env, push_async

from qevasion.loader import DATA_CACHE
# The official 237-row evaluation-phase test set, copied from the organizers' CSV.
# Row order is load-bearing: Codabench scores line i against row i. Verified
# identical, row by row, to the independent copy in TeleAI's eval_test.jsonl.
TEST_CSV = Path(__file__).resolve().parent / "data" / "clarity_task_evaluation_dataset.csv"
N_TEST = 237


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    if len(df) != N_TEST:
        raise SystemExit(f"{TEST_CSV} has {len(df)} rows, expected {N_TEST}")
    return df

# Fixed for every run so that the internal-validation split is identical across
# configurations and seeds. Comparing two configurations whose held-out sets
# differ measures the split as much as the configuration.
SPLIT_SEED = 12345
ANNOTATOR_IDS = ("85", "86", "89")


# --------------------------------------------------------------------------
# Input construction
# --------------------------------------------------------------------------


def build_inputs(df, mode: str) -> tuple[list[str], list[str]]:
    """Return (text_a, text_b) per row.

    The label is a judgment about whether *this sub-question* was answered, so
    the sub-question is always segment A and the answer always segment B; the
    segment boundary is the one piece of structure the model gets for free.

    `mode` controls how much of the surrounding turn is visible:
      qa      sub-question | answer                      (the minimal pair)
      qqa     sub-question [orig question] | answer      (adds the raw turn,
              which carries the other sub-questions and hence tells the model
              what the answer is *also* responding to -- the single largest
              source of confusion between Partial/half-answer and Explicit)
    """
    q = df["question"].astype(str).str.strip()
    a = df["interview_answer"].astype(str).str.strip()
    if mode == "qa":
        return q.tolist(), a.tolist()
    if mode == "full":
        # E8. The full journalist turn goes in segment A with the sub-question,
        # each labelled, so the model can see what ELSE the answer was responding
        # to (69% of rows share their answer with another sub-question). The
        # sub-question comes FIRST: ChulaNLP put it last, which means right-
        # truncation deletes it on exactly the long items where attribution is
        # hardest. Give segment A room (--a-budget 256 covers p95 of train).
        iq = df["interview_question"].astype(str).str.strip()
        return ("Sub-question: " + q + " Full question: " + iq).tolist(), a.tolist()
    if mode == "qqa":
        iq = df["interview_question"].astype(str).str.strip()
        return (q + " [CONTEXT] " + iq).tolist(), a.tolist()
    raise ValueError(f"unknown input mode {mode!r}")


def encode_pair(
    tok, text_a: str, text_b: str, max_len: int, a_budget: int, head_frac: float
) -> tuple[list[int], list[int]]:
    """Build `[CLS] a [SEP] b [SEP]` with head+tail truncation applied to b.

    29% of items exceed 512 tokens (reports/02_experiment_log.md, Step 0), so the
    truncation rule is a real modelling choice, not plumbing. Right-truncation
    discards the end of the answer, which is where a politician who eventually
    gets to the point gets to it. `head_frac < 1` keeps the first `head_frac`
    of the budget and the last `1 - head_frac`, following Sun et al. (CCL 2019),
    who found head+tail the best-performing truncation for long-document
    classification.
    """
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


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class ClarityEncoder(nn.Module):
    """Cross-encoder trunk with a configurable output head.

    `head`:
      flat        Linear(H, 9). The reference every other variant is measured
                  against, and the thing the user asked for first.
      hier        flat, plus an auxiliary Linear(H, 3) on the clarity level. The
                  organizers' overview reports that every system exploiting the
                  taxonomy's hierarchy beat those doing flat 9-way inference;
                  this is the cheapest possible instantiation of that -- the
                  9-way head still makes the prediction, the 3-way head only
                  shapes the representation.
      hsoft       E9. A trained hierarchy over the official taxonomy: one head scores
                  the 3 clarity levels, another the 9 leaves, and
                      p(leaf) = p(level) * p(leaf | level),
                  where p(leaf | level) is a softmax over only the leaves inside
                  that level. The head returns log p(leaf), so ordinary cross-
                  entropy on it is exactly the hierarchical loss
                      -log p(level of y) - log p(y | level of y):
                  every example trains the 3-way decision AND the choice within
                  its level, with no weighting to tune. Unlike `hier`, the coarse
                  head is part of the prediction, not just an auxiliary signal;
                  and Subtask 1 comes straight from p(level) = sum of its leaves.
                  `Clear Reply` has one leaf (Explicit), so p(Explicit) is simply
                  p(Clear Reply).
      annotator   flat, plus a per-annotator bias vector added to the logits
                  (Davani et al., TACL 2022, "annotator-level" modelling). The
                  train split carries `annotator_id`, so each row's label is a
                  judgment by a *known* person, and pooling them into one head
                  throws that away. At inference the three biased posteriors are
                  kept separately: the official scorer credits a prediction that
                  matches ANY annotator, so what it rewards is a model of the
                  panel, not of the consensus.
    """

    def __init__(self, name: str, head: str, dropout: float, pooling: str, n_ann: int = 3):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(name)
        # dtype=float32 is NOT a default -- it is load-bearing, and omitting it
        # silently destroys training on this stack.
        #
        # transformers 5.x loads a checkpoint in its OWN stored dtype; 4.x always
        # loaded fp32. The Hub weights for microsoft/deberta-v3-{base,large} are
        # stored in float16, so `AutoModel.from_pretrained(name)` now returns an
        # fp16 model. DeBERTa-v3's disentangled attention overflows in fp16:
        # HuggingFace's own DebertaV2ForSequenceClassification produced `nan`
        # loss from the first step, and with gradient clipping masking the nan it
        # instead sat at exactly the label-prior entropy (1.887), predicting one
        # class for every input. It could not overfit 128 examples in 12 epochs.
        # roberta-base, whose Hub weights are fp32, was unaffected -- which is
        # what isolated it to the checkpoint dtype rather than the recipe.
        #
        # Master weights stay fp32; bf16 is applied by autocast at compute time,
        # which keeps the speed without the range problem. (A `transformers` 5.x
        # behaviour change; see reports/02_experiment_log.md, Step 1.)
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        H = self.cfg.hidden_size
        self.head = head
        self.pooling = pooling
        # DeBERTa's own ContextPooler: dense + tanh on the [CLS] state. Kept for
        # mean pooling too so the two differ only in what is pooled.
        self.pooler = nn.Linear(H, H)
        self.drop = nn.Dropout(dropout)
        self.cls9 = nn.Linear(H, N_EVASION)
        self.cls3 = nn.Linear(H, 3) if head == "hier" else None
        self.branch = None
        if head == "hsoft":
            self.branch = nn.Linear(H, N_CLARITY)
            part = torch.as_tensor(OFFICIAL_PARTITION_MAP)
            for b in range(N_CLARITY):  # leaf indices inside each clarity level
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
            # log p(leaf) = log p(level) + log p(leaf | level); computed in fp32.
            log_level = F.log_softmax(self.branch(pooled).float(), dim=-1)
            z = logits.float()
            logp = torch.empty_like(z)
            for b in range(N_CLARITY):
                idx = getattr(self, f"members_{b}")
                logp[:, idx] = F.log_softmax(z[:, idx], dim=-1) + log_level[:, b:b + 1]
            logits = logp  # normalised: cross-entropy on it is the hierarchical NLL
        aux = self.cls3(pooled) if self.cls3 is not None else None
        return logits, aux

    def all_annotator_logits(self, pooled_logits: torch.Tensor) -> torch.Tensor:
        """(B, 9) unbiased logits -> (B, n_ann, 9) per-annotator logits."""
        return pooled_logits.unsqueeze(1) + self.ann_bias.unsqueeze(0)


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


def make_loss(kind: str, prior: np.ndarray, smoothing: float, device: str, gamma: float = 2.0):
    """Return `fn(logits, target) -> scalar`.

    The metric is macro-F1 over all nine classes with every class worth exactly
    1/9 regardless of frequency, while the rarest class is
    2.3% of train. Plain CE targets accuracy, which is the wrong quantity by
    construction, so the objective is an axis rather than a default.

      ce        cross-entropy, optionally label-smoothed. The reference.
      bal       Balanced Softmax (Ren et al., NeurIPS 2020): add log(prior) to
                the logits *inside* the loss. Fisher-consistent for the
                balanced error rate, and unlike inverse-frequency weighting it
                leaves the learned posterior interpretable -- at inference the
                term is simply dropped.
      invfreq   inverse-frequency class weights. Included because the overview
                report says this is what the encoder-based teams used, so it is
                the field's de facto baseline and belongs in the comparison.
      focal     focal loss (Lin et al., ICCV 2017): CE scaled by (1 - p_t)^gamma,
                which down-weights examples the model already gets right and so
                concentrates training on hard ones.
      bal_focal focal loss applied to balanced-softmax logits (z + log prior).
                The two corrections act on different things: Balanced Softmax
                removes the class prior from the learned posterior; focal
                reweights *examples* by difficulty regardless of class. Used in
                E8. As with `bal`, the prior term is dropped at inference.
    """
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


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------


def param_groups(model: ClarityEncoder, lr: float, head_lr: float, decay: float, wd: float):
    """Layer-wise learning-rate decay.

    Lower layers encode generic syntax that 3,448 examples cannot improve and
    can easily damage; the standard remedy is to scale the learning rate down
    with depth (Howard & Ruder, ACL 2018; used throughout the ELECTRA/DeBERTa
    fine-tuning recipes). With only 3.4k examples this matters more than usual
    -- fine-tuning instability on small data is a documented failure mode
    (Mosbach et al., ICLR 2021).
    """
    n_layers = model.cfg.num_hidden_layers
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")

    def wd_of(name: str) -> float:
        return 0.0 if any(k in name for k in no_decay) else wd

    def depth_of(name: str) -> int:
        # embeddings = 0, encoder layer i = i + 1, head = n_layers + 1
        if not name.startswith("enc."):
            return n_layers + 1
        if "embeddings" in name:
            return 0
        for part in name.split("."):
            if part.isdigit():
                return int(part) + 1
        return n_layers  # encoder-level extras (rel. embeddings, final LN)

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
    """Re-initialise the top `k` transformer layers.

    Zhang et al. (ICLR 2021) show the top layers of a pretrained encoder are
    specialised to the pretraining objective and that discarding them helps on
    small downstream sets. Cheap to test, so it is a flag rather than a belief.
    """
    if k <= 0:
        return
    layers = model.enc.encoder.layer
    for layer in layers[-k:]:
        layer.apply(model.enc._init_weights)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


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
    return np.concatenate(out, axis=0)


def inset_diagnostics(y_pred: np.ndarray, gold_mask: np.ndarray) -> dict:
    """In-set rate and class coverage -- the two terms macro-F1 decomposes into.

    See `reports/01_scorer_geometry.md`. When every prediction lands in its
    reference set, FP and FN are zero for every class, so

        macro-F1 = (number of distinct classes named) / 9

    exactly. Macro-F1 is therefore a monotone summary of two quantities, and
    reporting them separately says *which* one a change moved: a model can gain
    by hitting the set more often, or by naming a rare class it previously never
    named, and those call for completely different fixes. No published system on
    this task reports either.
    """
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
    """Collapse per-annotator probabilities to one posterior by averaging.

    Averaging probabilities (not logits) is the mixture the generative story
    implies: a training row is produced by one annotator drawn uniformly, so the
    marginal over an unknown annotator is the mean of their posteriors.
    """
    return probs.mean(axis=1) if probs.ndim == 3 else probs


# --------------------------------------------------------------------------
# Config and run
# --------------------------------------------------------------------------


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
    focal_gamma: float = 2.0  # Lin et al.'s default; used by --loss focal / bal_focal
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
    # Checkpoint selection. "best" = best internal-val epoch (the default and the
    # honest choice). "last" = final epoch, used in fold mode, where the held-out
    # fold must not also choose the epoch that produces its own predictions.
    select: str = "best"
    # Cross-fitting. fold >= 0 trains on the other n_folds-1 folds of the FULL
    # train split and predicts fold `fold` -- out-of-fold probabilities for every
    # training row, which the re-ranker needs as honest candidate sets.
    fold: int = -1
    n_folds: int = 5
    # Smoke testing only: cap the number of training rows.
    max_train: int = 0
    push_to_hub: bool = False


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


def stratified_val_index(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Stratified held-out index, identical for every run (SPLIT_SEED)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N_EVASION):
        rows = np.where(y == c)[0]
        rng.shuffle(rows)
        idx.extend(rows[: max(1, int(round(len(rows) * frac)))])
    return np.sort(np.asarray(idx))


def stratified_fold_index(y: np.ndarray, n_folds: int, fold: int, seed: int) -> np.ndarray:
    """Rows of fold `fold` under a stratified K-fold partition fixed by `seed`."""
    rng = np.random.default_rng(seed)
    assign = np.empty(len(y), dtype=np.int64)
    for c in range(N_EVASION):
        rows = np.where(y == c)[0]
        rng.shuffle(rows)
        assign[rows] = np.arange(len(rows)) % n_folds
    return np.sort(np.where(assign == fold)[0])


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

    if cfg.fold >= 0:
        val_idx = stratified_fold_index(y_train, cfg.n_folds, cfg.fold, SPLIT_SEED)
    else:
        val_idx = stratified_val_index(y_train, cfg.val_frac, SPLIT_SEED)
    tr_mask = np.ones(len(y_train), dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]
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
        y_train[tr_idx],
        c_train[tr_idx],
        ann_train[tr_idx],
        cfg,
    )
    ds_va = PairDataset(
        tok,
        [a_tr[i] for i in val_idx],
        [b_tr[i] for i in val_idx],
        y_train[val_idx],
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

    model = ClarityEncoder(cfg.model, cfg.head, cfg.dropout, cfg.pooling).to(device)
    reinit_top_layers(model, cfg.reinit_top)
    if cfg.grad_checkpoint:
        model.enc.gradient_checkpointing_enable()

    prior = np.bincount(y_train[tr_idx], minlength=N_EVASION).astype(np.float64)
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

    # Resume after a restart: everything that shapes the rest of the run is
    # restored, including the shuffle generator, so the remaining epochs see the
    # same data order they would have seen without the interruption.
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

        # Model selection uses the INTERNAL validation split only. The 308-item
        # dev set is the project's single multi-reference scoring surface and is
        # far too small to select on without overfitting it; it is scored each
        # epoch for monitoring and reported, never optimised against.
        val_pred = consensus(val_probs).argmax(1)
        val_score = macro_f1_single_label(val_pred, y_train[val_idx], EVASION_LABELS).macro_f1
        dev_pred = consensus(dev_probs).argmax(1)
        s2 = score_subtask2(dev_pred, dev_mask)
        s1 = score_subtask1(leaf_to_official_clarity(dev_pred), dev_clarity)

        diag = inset_diagnostics(dev_pred, dev_mask)
        row = {
            "epoch": epoch,
            "train_loss": running / max(nb, 1),
            "val_macro_f1": val_score,
            "dev_subtask2_macro_f1": s2.macro_f1,
            "dev_subtask1_macro_f1": s1.macro_f1,
            "dev_inset_rate": diag["inset_rate"],
            "dev_classes_named": diag["classes_named"],
            "minutes": (time.time() - t0) / 60,
        }
        history.append(row)
        print(
            f"[{cfg.name} {tag}] ep{epoch} loss={row['train_loss']:.4f} "
            f"val={val_score:.4f} devS2={s2.macro_f1:.4f} devS1={s1.macro_f1:.4f} "
            f"inset={diag['inset_rate']:.3f} named={diag['classes_named']}/9 "
            f"({row['minutes']:.1f}m)",
            flush=True,
        )
        tracker.log(row, step=epoch)
        if cfg.select == "last" or val_score > best["val_macro_f1"]:
            best = dict(row)
            best_dev_probs, best_val_probs, best_test_probs = dev_probs, val_probs, test_probs
            if cfg.save_model:
                # bf16 halves the checkpoint (1.7 -> 0.87 GB) and the upload; the
                # weights are only ever used for inference from here.
                atomic_torch_save(
                    {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                     for k, v in model.state_dict().items()},
                    outdir / "model.pt",
                )
        # Epoch-level checkpoint, written AFTER the best-model update so the two
        # can never disagree about which epoch is best.
        save_resume(outdir, epoch=epoch, model=model, opt=opt, sched=sched, gen=g, extra={
            "history": history, "best": best,
            "best_probs": (best_dev_probs, best_val_probs, best_test_probs),
            "elapsed_s": time.time() - t0,
        })

    dev_pred = consensus(best_dev_probs).argmax(1)
    s2 = score_subtask2(dev_pred, dev_mask)
    s1 = score_subtask1(leaf_to_official_clarity(dev_pred), dev_clarity)

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "dev_probs.npy", best_dev_probs)
    np.save(outdir / "val_probs.npy", best_val_probs)
    np.save(outdir / "val_index.npy", val_idx)
    np.save(outdir / "test_probs.npy", best_test_probs)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    result = {
        "config": asdict(cfg),
        "selected_epoch": best["epoch"],
        "val_macro_f1": best["val_macro_f1"],
        "dev_subtask2_macro_f1": s2.macro_f1,
        "dev_subtask1_macro_f1": s1.macro_f1,
        # The two terms macro-F1 decomposes into -- see reports/01_scorer_geometry.md.
        "dev_inset": inset_diagnostics(dev_pred, dev_mask),
        "dev_subtask2_per_class": {
            k: {"f1": v.f1, "support": v.support, "tp": v.tp, "fp": v.fp, "fn": v.fn}
            for k, v in s2.per_class.items()
        },
        "history": history,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "wall_minutes": (time.time() - t0) / 60,
    }
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
    clear_resume(outdir)  # the run is complete; metrics.json is now the record
    tracker.summary({
        "selected_epoch": result["selected_epoch"],
        "best_val_macro_f1": result["val_macro_f1"],
        "dev_subtask2_macro_f1": result["dev_subtask2_macro_f1"],
        "dev_subtask1_macro_f1": result["dev_subtask1_macro_f1"],
        "dev_inset_rate": result["dev_inset"]["inset_rate"],
        "peak_vram_gb": result["peak_vram_gb"],
    })
    tracker.finish()
    if cfg.push_to_hub:
        push_async(outdir)
    print(
        f"[{cfg.name} {tag}] DONE  devS2={s2.macro_f1:.4f} devS1={s1.macro_f1:.4f} "
        f"peak={result['peak_vram_gb']:.1f}GB  {result['wall_minutes']:.1f}m",
        flush=True,
    )
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
    p.add_argument("--outroot", default=os.environ.get("CLARITY_RUNS", str(Path(__file__).resolve().parent / "runs")))
    args = p.parse_args()
    cfg = Cfg(**{k: v for k, v in vars(args).items() if k != "outroot"})
    load_env()  # HF_TOKEN also authenticates model downloads
    sub = f"fold{cfg.fold}" if cfg.fold >= 0 else f"seed{cfg.seed}"
    outdir = Path(args.outroot) / cfg.name / sub
    outdir.mkdir(parents=True, exist_ok=True)
    run(cfg, outdir)


if __name__ == "__main__":
    main()
