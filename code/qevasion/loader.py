"""QEvasion data layer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from qevasion.labels import (
    EVASION_LABELS,
    OOV_INDEX,
    encode_clarity,
    encode_evasion,
    multi_reference_mask,
    normalize_evasion,
)

__all__ = [
    "QEvasionSplits",
    "load_qevasion",
    "load_github_multireference",
    "derive_dev_consensus_leaf",
    "annotator_marginals",
    "build_text_pairs",
    "write_frozen_splits",
    "load_frozen_splits",
]

HF_REPO: Final[str] = "ailsntua/QEvasion"
HF_TRAIN_URL: Final[str] = (
    f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/train-00000-of-00001.parquet"
)
HF_DEV_URL: Final[str] = (
    f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/test-00000-of-00001.parquet"
)

ANNOTATOR_COLUMNS: Final[tuple[str, str, str]] = ("annotator1", "annotator2", "annotator3")

IDENTITY_COLUMNS: Final[tuple[str, str, str]] = (
    "interview_question",
    "question",
    "interview_answer",
)


@dataclass(frozen=True)
class QEvasionSplits:
    """The canonical splits, with names that say what they actually are."""

    train: pd.DataFrame
    """3,448 rows. Single `evasion_label` plus `annotator_id`."""

    dev: pd.DataFrame
    """308 rows. Multi-reference `annotator1/2/3`; NO `evasion_label`.
    Published on HuggingFace under the misleading split name `test`."""

    def __post_init__(self) -> None:
        if "evasion_label" not in self.train.columns:
            raise ValueError("train split is missing `evasion_label`")
        for col in ANNOTATOR_COLUMNS:
            if col not in self.dev.columns:
                raise ValueError(f"dev split is missing `{col}`")


DATA_CACHE: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "cache"


def load_qevasion(cache_dir: str | Path | None = DATA_CACHE) -> QEvasionSplits:
    """Load the canonical HuggingFace parquet splits."""
    if cache_dir is not None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        train_path, dev_path = cache / "train.parquet", cache / "dev.parquet"
        if not train_path.exists():
            pd.read_parquet(HF_TRAIN_URL).to_parquet(train_path)
        if not dev_path.exists():
            pd.read_parquet(HF_DEV_URL).to_parquet(dev_path)
        train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    else:
        train, dev = pd.read_parquet(HF_TRAIN_URL), pd.read_parquet(HF_DEV_URL)

    return QEvasionSplits(train=train.reset_index(drop=True), dev=dev.reset_index(drop=True))


def load_github_multireference(ref_root: str | Path) -> pd.DataFrame:
    """Load the organizers' RAW per-annotator CSV as a cross-check."""
    path = Path(ref_root) / "Question-Evasion" / "dataset" / "Inter-Annotator" / "test_set.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"organizers' raw multi-reference CSV not found at {path}; "
            "run scripts/clone_refs.sh first"
        )
    df = pd.read_csv(path, engine="python")

    def _strip_prefix(value: object) -> str:
        """'2.3 Partial/half-answer' -> 'Partial/half-answer'."""
        text = str(value).strip()
        head, _, tail = text.partition(" ")
        return tail.strip() if head[:1].isdigit() and "." in head else text

    for col in ANNOTATOR_COLUMNS:
        source = col.capitalize()
        if source in df.columns:
            df[col] = df[source].map(_strip_prefix).map(normalize_evasion)

    df["n_out_of_taxonomy"] = sum(
        (~df[col].isin(EVASION_LABELS)).astype(int) for col in ANNOTATOR_COLUMNS
    )
    return df


def dev_reference_mask(dev: pd.DataFrame) -> np.ndarray:
    """(308, 9) boolean reference-set mask for the multi-reference scorer."""
    refs = [
        [row[col] for col in ANNOTATOR_COLUMNS]
        for _, row in dev[list(ANNOTATOR_COLUMNS)].iterrows()
    ]
    return multi_reference_mask(refs)


def derive_dev_consensus_leaf(dev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote leaf label for the dev split, which has no `evasion_label`."""
    leaf_idx = np.full(len(dev), OOV_INDEX, dtype=np.int64)
    has_majority = np.zeros(len(dev), dtype=bool)

    for i, (_, row) in enumerate(dev[list(ANNOTATOR_COLUMNS)].iterrows()):
        labels = [normalize_evasion(str(row[c])) for c in ANNOTATOR_COLUMNS]
        (top, count), = Counter(labels).most_common(1)
        if count > 1:
            leaf_idx[i] = EVASION_LABELS.index(top) if top in EVASION_LABELS else OOV_INDEX
            has_majority[i] = leaf_idx[i] != OOV_INDEX

    return leaf_idx, has_majority


def consensus_level(dev: pd.DataFrame) -> np.ndarray:
    """(N,) number of DISTINCT labels per item: 1 unanimous, 2, or 3."""
    return np.asarray(
        [
            len({normalize_evasion(str(row[c])) for c in ANNOTATOR_COLUMNS})
            for _, row in dev[list(ANNOTATOR_COLUMNS)].iterrows()
        ],
        dtype=np.int64,
    )


def annotator_marginals(train: pd.DataFrame) -> pd.DataFrame:
    """Per-annotator label marginals from the training split: rows sum to 1."""
    return (
        pd.crosstab(train["annotator_id"], train["evasion_label"], normalize="index")
        .reindex(columns=list(EVASION_LABELS))
        .fillna(0.0)
    )


def build_text_pairs(df: pd.DataFrame) -> list[str]:
    """Concatenate [sub-question ; answer], the model input for both subtasks."""
    return [
        f"{str(q).strip()} {str(a).strip()}"
        for q, a in zip(df["question"], df["interview_answer"], strict=True)
    ]


def _hash_rows(df: pd.DataFrame) -> str:
    """Content hash over the identity columns, stable across row order."""
    keys = sorted(
        "\x1f".join(str(row[c]) for c in IDENTITY_COLUMNS)
        for _, row in df[list(IDENTITY_COLUMNS)].iterrows()
    )
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def write_frozen_splits(splits: QEvasionSplits, path: str | Path) -> dict:
    """Write the frozen split artifact every later phase reads."""
    artifact = {
        "schema_version": 1,
        "source": HF_REPO,
        "note": (
            "The HuggingFace split named 'test' is the shared task's VALIDATION "
            "set (308 rows). The real 237-row evaluation test set is not "
            "published on HuggingFace."
        ),
        "splits": {
            "train": {
                "n_rows": int(len(splits.train)),
                "content_hash": _hash_rows(splits.train),
                "has_evasion_label": True,
                "has_multi_reference": False,
            },
            "dev": {
                "n_rows": int(len(splits.dev)),
                "content_hash": _hash_rows(splits.dev),
                "has_evasion_label": False,
                "has_multi_reference": True,
            },
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def load_frozen_splits(path: str | Path) -> dict:
    """Read the frozen split artifact."""
    return json.loads(Path(path).read_text())


def verify_against_frozen(splits: QEvasionSplits, path: str | Path) -> None:
    """Raise if the loaded data no longer matches the frozen artifact."""
    artifact = load_frozen_splits(path)
    for name, frame in (("train", splits.train), ("dev", splits.dev)):
        expected = artifact["splits"][name]
        if len(frame) != expected["n_rows"]:
            raise ValueError(
                f"{name}: row count {len(frame)} != frozen {expected['n_rows']}"
            )
        actual = _hash_rows(frame)
        if actual != expected["content_hash"]:
            raise ValueError(
                f"{name}: content hash {actual[:12]} != frozen "
                f"{expected['content_hash'][:12]}; the upstream data changed"
            )


def encode_labels(splits: QEvasionSplits) -> dict[str, np.ndarray]:
    """Integer-encode every label column the project uses."""
    return {
        "train_leaf": encode_evasion(splits.train["evasion_label"].tolist()),
        "train_clarity": encode_clarity(splits.train["clarity_label"].tolist()),
        "dev_clarity": encode_clarity(splits.dev["clarity_label"].tolist()),
        "dev_reference_mask": dev_reference_mask(splits.dev),
        "dev_consensus_level": consensus_level(splits.dev),
    }
