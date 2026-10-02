#!/usr/bin/env python3
"""Per-class counts for the report's taxonomy table (Table 1, sec:task)."""

from __future__ import annotations

import numpy as np

from qevasion.labels import EVASION_LABELS, OFFICIAL_CLARITY_OF_LEAF, encode_evasion
from qevasion.loader import dev_reference_mask, load_qevasion

sp = load_qevasion()
y = encode_evasion(sp.train["evasion_label"].tolist())
train_n = np.bincount(y, minlength=len(EVASION_LABELS))
dev_sets = dev_reference_mask(sp.dev).sum(0)

print(f"train rows {len(y)}, dev items {len(sp.dev)}")
print(f"{'label':<22} {'clarity':<16} {'train n':>8} {'train %':>8} {'dev ref sets':>13}")
for c, lab in enumerate(EVASION_LABELS):
    print(f"{lab:<22} {OFFICIAL_CLARITY_OF_LEAF[lab]:<16} {train_n[c]:>8} {100 * train_n[c] / len(y):>7.1f}% {dev_sets[c]:>13}")
