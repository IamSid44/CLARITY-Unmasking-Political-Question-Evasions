# Audit — what exists, checked on 2026-10-02

## Environment

| | |
|---|---|
| machine | lab server: 64 CPU cores, 503 GB RAM, 2× NVIDIA RTX PRO 6000 Blackwell (96 GB each), shared with other users; the second card is split into two 48 GB slices (MIG) |
| software | Python 3.12.3, torch 2.11.0+cu128, transformers 5.17.0, peft 0.20.0, scikit-learn 1.9.1, numpy 2.5.2, pandas 3.0.5 |
| cloud | JarvisLabs, about ₹5,880 of credit, reserved for the LLM runs |

## Data, verified

| fact | value |
|---|---|
| splits | train 3,448 · dev 308 · test 237 (test labels not available anywhere; Codabench closed) |
| train labels | one per row, with `annotator_id` (85: 1,102 rows; 86: 1,290; 89: 1,056) |
| dev labels | three annotators per row (`annotator1–3`); majority exists for 275 items, 33 three-way splits |
| dev consensus | unanimous 125, 2–1 150, all different 33 |
| inputs over 512 tokens ([sub-question; answer]) | 101 of 308 dev items |
| full input (sub-question + full question + answer), median tokens | train 358, dev 474, **test 137** |

## Status checklist (the team brief's §2.4)

| # | item | status | evidence |
|---|---|---|---|
| 1 | data loading | done | `clarity/qevasion/loader.py` |
| 2 | multi-reference scorer, tested | done | `clarity/qevasion/scoring.py`; `clarity/verify_scorer_geometry.py`; `clarity/reports/01_scorer_geometry.md` |
| 3 | frozen splits (train-internal slice, dev-A/B) | done | `splits/train_internal_val_index.json` (345 rows, identical across all 45 nine-way runs); `splits/dev_A_B.json` |
| 4 | trivial baselines | done | `analysis/trivial_baselines.csv` |
| 5 | encoder baseline, 9-way → derived 3-way | done, **DeBERTa-v3-large** (not base) | `L0_large_base`, `E11_base_8ep`; log §E0 |
| 6 | direct 3-way control | **missing** | planned, Phase D |
| 7 | multiple seeds | done, 10 seeds per main configuration | log §E11 |
| 8 | DeBERTa-v3-large | done (it is the backbone throughout) | — |
| 9 | decision rule (temperature, per-class bias, logit adjustment) | done | log §E4; `analysis/main_table.csv`; RESULTS §1 and §7 |
| 10 | error analysis | done | `analysis/` (confusion, per-class, consensus, length, confidence, examples) |
| 11 | chunk selection + random control (C2 re-scoped) | not done; replaced | the full-question input addresses attribution (+0.022 per model); only 20 dev items are still cut, and test inputs are short |
| 12 | coverage auxiliary head (C1 re-scoped) | not done | three other hierarchy forms tested, all negative (log §E5, §E9, §E12b); C2 found coverage is not the axis annotators agree on |
| 13 | annotator-ID check (C3 feasibility) | done | train has annotator IDs, one per item; AUDIT above; `analysis/mideval_analysis.txt` §A |
| 14 | LLM teacher pilot | planned, Phase C | — |
| 15 | cascade / deferral curve | planned, Phase B; groundwork done (confidence and top-k analysis) | RESULTS §6 |
| 16 | compression and latency | planned, Phase E | — |
| — | Qwen3-8B LoRA classifier (E13) | **done at 10 seeds** (2026-10-02): single S2 0.476 / S1 0.709, 10/10 seeds above DeBERTa; system S2 0.543 / S1 0.746 vs 0.405 / 0.648; 12 epochs adopted on the slice (3-seed screen); all of train not distinguishable | RESULTS §9; REPORT §7; `clarity/reports/raw/E13_final_analysis.txt` |

## Known issues

- The repository's git remote URL contains a GitHub personal access token in plain text. Revoke
  it and use a credential helper or SSH.
- `clarity/run_queue.sh` and `run_pipeline.sh` hard-code the lab server's Python path.
- `clarity/reports/raw/E11_replication.txt` contains hand-appended tables that no script
  regenerates (see REPRODUCE.md).
