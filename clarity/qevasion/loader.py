"""QEvasion data layer.

Canonical source is the HuggingFace parquet (`ailsntua/QEvasion`). The organizers' GitHub CSVs are loaded separately as a CROSS-CHECK
only -- they are the raw pre-cleanup annotations and disagree with the published
release in ways that matter (see `load_github_multireference`).

Split structure, verified empirically and not taken from prose
--------------------------------------------------------------
The HuggingFace release publishes two splits under the names `train` and `test`,
but the 308-row `test` split is the shared task's VALIDATION set, not its test
set. The organizers' overview paper (Table 1) gives train 3,448 / validation 308
/ test 237, and the real 237-row evaluation set is not on HuggingFace at all.

    train (3,448)  evasion_label YES | annotator_id YES (85/86/89)
                   annotator1/2/3 EMPTY
    dev   (308)    evasion_label EMPTY | annotator_id EMPTY
                   annotator1/2/3 POPULATED (multi-reference gold)
                   clarity_label YES

This asymmetry drives the whole project's design:

  * `annotator_id` on train gives per-annotator MARGINALS -> contribution C3.
  * `annotator1/2/3` on dev gives multi-reference REFERENCE SETS -> C4, and the
    three independent judgments the C2 agreement test re-analyzes.
  * dev has NO `evasion_label`, so a single dev leaf gold must be DERIVED. A
    majority exists for only 275/308 items; the remaining 33 are three-way
    splits the organizers' expert adjudicated and whose adjudication was not
    released. See `derive_dev_consensus_leaf`.

`Implementation_Guide.md` Key Finding #1 states the public release has empty
annotator columns and that C3/C4 are therefore blocked. That is true only of the
train split, and the guide's conclusion does not hold: the multi-reference labels
are public and complete.
"""

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

# The columns that define item identity for deduplication. A single interview
# answer is paired with several decomposed sub-questions, so the sub-question
# must be part of the key -- deduplicating on the answer alone would collapse
# genuinely distinct items.
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


# The canonical splits are cached here; downloaded from HuggingFace if absent.
DATA_CACHE: Final[Path] = Path(__file__).resolve().parent.parent / "data" / "cache"


def load_qevasion(cache_dir: str | Path | None = DATA_CACHE) -> QEvasionSplits:
    """Load the canonical HuggingFace parquet splits.

    The HF split named `test` is returned as `dev`, because that is what it is.
    """
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
    """Load the organizers' RAW per-annotator CSV as a cross-check.

    Read-only reference data, NOT the canonical source. It differs from the
    published release in two ways that must never leak into an experiment:

      * 317 rows rather than 308 -- the pre-cleanup count.
      * ELEVEN label values rather than nine. It contains `2.9 Diffusion` and
        `2.5 Contradictory`, which are outside the SemEval taxonomy, plus the
        `N.N ` numeric prefixes that the published release strips.

    A `n_out_of_taxonomy` column is attached so the audit can quantify the
    317 -> 308 delta rather than silently dropping rows.
    """
    path = Path(ref_root) / "Question-Evasion" / "dataset" / "Inter-Annotator" / "test_set.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"organizers' raw multi-reference CSV not found at {path}; "
            "run scripts/clone_refs.sh first"
        )
    # The C engine fails on embedded newlines inside quoted interview answers.
    df = pd.read_csv(path, engine="python")

    def _strip_prefix(value: object) -> str:
        """'2.3 Partial/half-answer' -> 'Partial/half-answer'."""
        text = str(value).strip()
        head, _, tail = text.partition(" ")
        return tail.strip() if head[:1].isdigit() and "." in head else text

    for col in ANNOTATOR_COLUMNS:
        source = col.capitalize()  # 'Annotator1' in the raw file
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
    """Majority-vote leaf label for the dev split, which has no `evasion_label`.

    Returns
    -------
    (leaf_idx, has_majority)
        `leaf_idx[i]` is the majority leaf index, or `OOV_INDEX` where all three
        annotators disagreed. `has_majority[i]` is the corresponding boolean.

    A majority exists for roughly 275/308 items; the other ~33 (10.7%) are
    three-way splits. The organizers resolved those with an expert adjudicator
    but did not release the adjudications, so those items have NO recoverable
    single gold leaf.

    This is a real constraint on Phase 6, not a detail: the C4 majority-vote
    control is UNDEFINED on those items and must be reported on the 275-item
    subset with the exclusion stated explicitly. It does not affect the
    multi-reference scorer itself, which needs only the reference SET and is
    perfectly well-defined on all 308.
    """
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
    """(N,) number of DISTINCT labels per item: 1 unanimous, 2, or 3.

    Phase 6 stratifies every C4 result by this. The pre-registered prediction is
    that gains fall on levels 2 and 3 and are absent at level 1 -- where the
    reference set is a singleton and the set-membership rule provably coincides
    with consensus selection.
    """
    return np.asarray(
        [
            len({normalize_evasion(str(row[c])) for c in ANNOTATOR_COLUMNS})
            for _, row in dev[list(ANNOTATOR_COLUMNS)].iterrows()
        ],
        dtype=np.int64,
    )


def annotator_marginals(train: pd.DataFrame) -> pd.DataFrame:
    """Per-annotator label marginals from the training split: rows sum to 1.

    Contribution C3's identifying assumption is that training items were
    distributed across annotators AT RANDOM, so differences between these
    marginals are annotator effects rather than item effects. Phase 5 validates
    the LEARNED marginal-bias vectors against this directly measured table; they
    should agree, and a disagreement means the model is wrong before it is used.
    """
    return (
        pd.crosstab(train["annotator_id"], train["evasion_label"], normalize="index")
        .reindex(columns=list(EVASION_LABELS))
        .fillna(0.0)
    )


def build_text_pairs(df: pd.DataFrame) -> list[str]:
    """Concatenate [sub-question ; answer], the model input for both subtasks.

    The sub-question (`question`) is the LLM-decomposed single-part question; the
    answer (`interview_answer`) is the FULL original answer, not a segment of it.
    That asymmetry is the whole reason C5 exists: locating which span of the
    answer -- if any -- responds to this particular sub-question is an unsolved
    sub-problem, not a given.
    """
    return [
        f"{str(q).strip()} {str(a).strip()}"
        for q, a in zip(df["question"], df["interview_answer"], strict=True)
    ]


# --- Frozen split artifact -------------------------------------------------


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
    """Write the frozen split artifact every later phase reads.

    Splits are NEVER re-derived on the fly. The hash makes an accidental change
    to the underlying data loud rather than silent: if the upstream dataset is
    revised, `load_frozen_splits` will refuse to match and the discrepancy
    surfaces immediately instead of quietly changing results mid-project.
    """
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
