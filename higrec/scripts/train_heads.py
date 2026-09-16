#!/usr/bin/env python3
"""Train every output-head variant on cached frozen features.

This is the C1 experiment (P4.1/P4.2) plus the C3 annotator model, run in the
frozen-trunk regime described in reasoning.md D-02.

Variants
--------
flat            Linear(H, 9). THE FROZEN REFERENCE everything is measured against.
factorized      FactorizedHead, marginal objective. C1.
factorized_sum  FactorizedHead, attribute-sum objective. Objective ablation.
param_matched   Capacity control. The factorized head has MORE parameters than
                flat (13 output units vs 9), so a naive flat-vs-factorized
                comparison confounds structure with capacity. This control
                matches the parameter count.
random_code     THE DECISIVE CONTROL (P4.2 step 5). Same factorized machinery, but
                the code table's rows are permuted, destroying semantics while
                preserving uniqueness and per-attribute value balance. If this
                matches `factorized`, the mechanism is regularization, not
                structure, and C1 as argued is dead.
annotator_bias  C3. Shared latent head plus a per-annotator bias vector added to
                the logits, trained on train where annotator_id exists. At
                inference the bias is dropped to recover the clean posterior; the
                three biased posteriors are what C4's q(c|x) consumes.

Protocol: 5-fold CV on train x N seeds. Dev is scored with the multi-reference
scorer. Per-example dev probability vectors are saved for every variant and seed,
because C4 runs post-hoc on them at zero additional cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from higrec.data.labels import EVASION_LABELS, N_EVASION, encode_evasion
from higrec.data.loader import dev_reference_mask, load_qevasion
from higrec.models.factorized import (
    ATTRIBUTE_SIZES,
    FactorizedConfig,
    FactorizedHead,
    random_code_index,
)
from higrec.scoring.official import macro_f1_multireference

VARIANTS = (
    "flat", "factorized", "factorized_sum", "param_matched",
    "random_code", "annotator_bias",
)


class FlatHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, N_EVASION)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x), dim=-1)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(self(x), y)


class ParamMatchedHead(nn.Module):
    """Flat head with a bottleneck sized to match the factorized parameter count.

    Factorized is H*13 + 13. This is H*k + k + k*9 + 9 with k = 13, giving
    H*13 + 13 + 126 -- within ~1% at H = 1024. Exact counts are logged so the
    residual difference is visible rather than assumed away.
    """

    def __init__(self, hidden: int, bottleneck: int = 13) -> None:
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, N_EVASION)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.up(self.down(x)), dim=-1)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(self(x), y)


class AnnotatorBiasHead(nn.Module):
    """C3: shared latent logits plus a per-annotator bias vector."""

    def __init__(self, hidden: int, n_annotators: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, N_EVASION)
        self.bias = nn.Parameter(torch.zeros(n_annotators, N_EVASION))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean, annotator-free posterior -- the bias is DROPPED at inference."""
        return F.log_softmax(self.linear(x), dim=-1)

    def observed(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x) + self.bias[a], dim=-1)

    def loss(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(self.observed(x, a), y)

    def all_annotator_probs(self, x: torch.Tensor) -> torch.Tensor:
        """(N, n_annotators, 9) -- exactly what C4's q(c|x) consumes."""
        return F.softmax(self.linear(x).unsqueeze(1) + self.bias.unsqueeze(0), dim=-1)


def build(variant: str, hidden: int, seed: int, n_annotators: int) -> nn.Module:
    if variant == "flat":
        return FlatHead(hidden)
    if variant == "param_matched":
        return ParamMatchedHead(hidden)
    if variant == "annotator_bias":
        return AnnotatorBiasHead(hidden, n_annotators)
    if variant == "factorized":
        return FactorizedHead(FactorizedConfig(hidden_size=hidden, objective="marginal"))
    if variant == "factorized_sum":
        return FactorizedHead(
            FactorizedConfig(hidden_size=hidden, objective="attribute_sum")
        )
    if variant == "random_code":
        head = FactorizedHead(FactorizedConfig(hidden_size=hidden, objective="marginal"))
        # Permuted rows: uniqueness and per-attribute balance preserved, semantics destroyed.
        head.code_index.copy_(random_code_index(seed=1000 + seed))
        return head
    raise ValueError(f"unknown variant {variant!r}")


def train_one(
    variant: str, Xtr: torch.Tensor, ytr: torch.Tensor, atr: torch.Tensor,
    hidden: int, seed: int, n_annotators: int, epochs: int, lr: float,
    weight_decay: float, device: str,
) -> nn.Module:
    torch.manual_seed(seed)
    model = build(variant, hidden, seed, n_annotators).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xtr.shape[0]
    batch = 256
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            if variant == "annotator_bias":
                loss = model.loss(Xtr[idx], ytr[idx], atr[idx])
            else:
                loss = model.loss(Xtr[idx], ytr[idx])
            loss.backward()
            opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="runs/features/deberta-v3-large_len512")
    ap.add_argument("--out-dir", default="runs/heads")
    ap.add_argument("--cache-dir", default="runs/data_cache")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    feat_dir = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr_all = np.load(feat_dir / "train.npy")
    Xdev_all = np.load(feat_dir / "dev.npy")
    hidden = Xtr_all.shape[1]
    print(f"[features] train {Xtr_all.shape} dev {Xdev_all.shape} from {feat_dir}")

    splits = load_qevasion(cache_dir=args.cache_dir)
    ytr_all = encode_evasion(splits.train["evasion_label"].tolist())
    annot_ids = sorted(splits.train["annotator_id"].unique().tolist())
    atr_all = np.array([annot_ids.index(a) for a in splits.train["annotator_id"]],
                       dtype=np.int64)
    dev_mask = dev_reference_mask(splits.dev)
    print(f"[labels] train {ytr_all.shape}, annotators {annot_ids}, "
          f"dev reference mask {dev_mask.shape}, mean |G| {dev_mask.sum(1).mean():.3f}")

    Xtr_t = torch.from_numpy(Xtr_all).float().to(device)
    ytr_t = torch.from_numpy(ytr_all).long().to(device)
    atr_t = torch.from_numpy(atr_all).long().to(device)
    Xdev_t = torch.from_numpy(Xdev_all).float().to(device)

    results: list[dict] = []
    started = time.time()

    for variant in args.variants:
        param_count = sum(p.numel() for p in
                          build(variant, hidden, 0, len(annot_ids)).parameters())
        print(f"\n=== {variant} ({param_count:,} trainable head params) ===")
        dev_probs_by_seed = []

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            fold_of = rng.permutation(len(ytr_all)) % args.folds
            cv_scores = []

            for fold in range(args.folds):
                tr = fold_of != fold
                va = ~tr
                model = train_one(variant, Xtr_t[tr], ytr_t[tr], atr_t[tr], hidden,
                                  seed * 100 + fold, len(annot_ids), args.epochs,
                                  args.lr, args.weight_decay, device)
                model.eval()
                with torch.no_grad():
                    pred = model(Xtr_t[va]).argmax(dim=-1).cpu().numpy()
                cv_scores.append(float((pred == ytr_all[va]).mean()))

            # Full-train model for the dev evaluation.
            model = train_one(variant, Xtr_t, ytr_t, atr_t, hidden, seed,
                              len(annot_ids), args.epochs, args.lr,
                              args.weight_decay, device)
            model.eval()
            with torch.no_grad():
                dev_log = model(Xdev_t)
                dev_p = dev_log.exp().cpu().numpy()
                if variant == "annotator_bias":
                    np.save(out_dir / f"annotator_probs_seed{seed}.npy",
                            model.all_annotator_probs(Xdev_t).cpu().numpy())
                    np.save(out_dir / f"annotator_bias_seed{seed}.npy",
                            model.bias.detach().cpu().numpy())
            dev_probs_by_seed.append(dev_p)

            dev_pred = dev_p.argmax(axis=1)
            score = macro_f1_multireference(dev_pred, dev_mask, EVASION_LABELS)
            in_set = float(dev_mask[np.arange(len(dev_pred)), dev_pred].mean())

            results.append({
                "variant": variant, "seed": seed, "params": param_count,
                "cv_accuracy": float(np.mean(cv_scores)),
                "dev_macro_f1": score.macro_f1,
                "dev_in_set_rate": in_set,
                "per_class_f1": {c: score.per_class[c].f1 for c in EVASION_LABELS},
            })
            print(f"  seed {seed}: cv_acc={np.mean(cv_scores):.4f} "
                  f"dev_macroF1={score.macro_f1:.4f} in_set={in_set:.3f}")

        np.save(out_dir / f"dev_probs_{variant}.npy", np.stack(dev_probs_by_seed))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[done] {time.time() - started:.1f}s -> {out_dir}")

    print("\n=== SUMMARY (dev, multi-reference macro-F1) ===")
    for variant in args.variants:
        rows = [r for r in results if r["variant"] == variant]
        f1 = np.array([r["dev_macro_f1"] for r in rows])
        acc = np.array([r["cv_accuracy"] for r in rows])
        ins = np.array([r["dev_in_set_rate"] for r in rows])
        print(f"  {variant:16s} macroF1 {f1.mean():.4f} +/- {f1.std():.4f} | "
              f"cv_acc {acc.mean():.4f} | in_set {ins.mean():.3f} | "
              f"{rows[0]['params']:,} params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
