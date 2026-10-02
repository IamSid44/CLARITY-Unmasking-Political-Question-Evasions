#!/usr/bin/env python3
"""E13 seeds 0-2: top-k coverage, confidence vs correctness, and prediction counts of the
3-seed probability average (argmax). Output: clarity/reports/raw/E13_Q8_topk_confidence.txt.
Exploratory (no dev choice depends on it). Run from the repo root:  python clarity/e13_topk.py
"""
import sys
import numpy as np
sys.path.insert(0, "clarity")
from qevasion.loader import dev_reference_mask, load_qevasion, DATA_CACHE
from qevasion.labels import EVASION_LABELS, encode_evasion
from qevasion.scoring import score_subtask2
sp = load_qevasion(DATA_CACHE); mask = dev_reference_mask(sp.dev)
ytr = encode_evasion(sp.train["evasion_label"].tolist())
P = [np.load(f"clarity/runs/Q8_fullq_lora/seed{s}/dev_probs.npy") for s in range(3)]
V = [(np.load(f"clarity/runs/Q8_fullq_lora/seed{s}/val_probs.npy"), np.load(f"clarity/runs/Q8_fullq_lora/seed{s}/val_index.npy")) for s in range(3)]
def topk(p, m, k): o = np.argsort(-p, 1)[:, :k]; return np.mean([m[i, o[i]].any() for i in range(len(p))])
print("dev: acceptable label within top-k (mean over 3 seeds)")
for k in (1, 2, 3, 5): print(f"  k={k}: {np.mean([topk(p, mask, k) for p in P]):.3f}")
print("train slice (one label): gold within top-k")
for k in (1, 2, 3, 5): print(f"  k={k}: {np.mean([np.mean([y in np.argsort(-p[i])[:k] for i, y in enumerate(ytr[vi])]) for p, vi in V]):.3f}")
print("dev in-set rate by confidence fifth (least -> most confident), mean over seeds")
rows = []
for p in P:
    c = p.max(1); pr = p.argmax(1); hit = mask[np.arange(len(p)), pr]; q = np.argsort(c)
    rows.append([hit[q[i * len(p) // 5:(i + 1) * len(p) // 5]].mean() for i in range(5)])
print("  " + "  ".join(f"{v:.2f}" for v in np.mean(rows, 0)))
E = np.mean(P, 0); print(f"3-seed probability average, argmax S2: {score_subtask2(E.argmax(1), mask).macro_f1:.4f} (singles {[round(score_subtask2(p.argmax(1), mask).macro_f1,3) for p in P]})")
print("3-seed probability average, argmax: how often each class is predicted vs how many dev reference sets contain it:")
pr = E.argmax(1); 
for j, n in enumerate(EVASION_LABELS): print(f"  {n:22s} predicted {int((pr==j).sum()):3d}  in reference sets {int(mask[:, j].sum()):3d}")
