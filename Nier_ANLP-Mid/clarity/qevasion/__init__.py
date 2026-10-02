"""Data layer, label vocabulary and official-scorer replica for QEvasion.

Copied from the team's analysis track (`higrec/src/higrec/`, now archived) so that
this repository is self-contained:

    labels.py   <- higrec/data/labels.py      label strings, encodings, the 9->3 map
    loader.py   <- higrec/data/loader.py      HuggingFace parquet splits (train / dev)
    scoring.py  <- higrec/scoring/official.py multi-reference macro-F1 (Subtask 2)
                                              and single-label macro-F1 (Subtask 1)

Only the import paths and the default data location were changed. The scorer was
conformance-tested in higrec against TeleAI's replica of the official Codabench
scorer (the only independent implementation available); clarity's copy was
checked to give identical results before the original was archived.
"""
