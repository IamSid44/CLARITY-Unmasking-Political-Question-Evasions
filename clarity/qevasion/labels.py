"""Canonical label vocabulary for SemEval-2026 Task 6 (CLARITY).

This module is the single source of truth for label strings, their integer
encodings, and the OFFICIAL 3-way clarity partition. Every other module imports
from here rather than hard-coding label strings, so that a change to the
vocabulary is impossible to make inconsistently.

Provenance
----------
The nine evasion labels and three clarity labels are the organizers' own
vocabulary, verified directly against the published dataset
(`ailsntua/QEvasion`), not taken from any participant's code.

The OFFICIAL partition below was verified ARITHMETICALLY against the training
split rather than copied from prose (see reports/03_data_audit.md):

    clarity_label counts : Clear Reply 1052 | Ambivalent 2040 | Clear Non-Reply 356
    evasion_label counts : Explicit 1052
                           Declining 145 + Claims ignorance 119 + Clarification 92 = 356

Those identities hold exactly, which pins the mapping with no ambiguity. It was
cross-checked a second way: on the 308-row multi-reference split, the published
`clarity_label` equals the majority vote of the three annotators' leaf labels
pushed through this partition on 306/308 items, the other 2 being three-way ties
the organizers' expert adjudicated.

Index order
-----------
`EVASION_LABELS` and `CLARITY_LABELS` are ordered to match the reference scorer's
declaration order. Index order is load-bearing: it fixes the column order of
every probability matrix, confusion matrix and per-class table in the project.
Do not reorder these tuples.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# --- Subtask 2: the nine evasion leaves -----------------------------------

EVASION_LABELS: Final[tuple[str, ...]] = (
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
)
N_EVASION: Final[int] = len(EVASION_LABELS)

# --- Subtask 1: the three clarity levels ----------------------------------

CLARITY_LABELS: Final[tuple[str, ...]] = (
    "Ambivalent",
    "Clear Non-Reply",
    "Clear Reply",
)
N_CLARITY: Final[int] = len(CLARITY_LABELS)

# --- The OFFICIAL 3-way partition -----------------------------------------
#
# Note the asymmetry that motivates the entire C2 contribution: Clear Reply
# holds exactly ONE leaf while Ambivalent holds FIVE spanning full, partial and
# zero information delivery.

OFFICIAL_CLARITY_OF_LEAF: Final[dict[str, str]] = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

# Sentinel for a label string that is not in the official vocabulary.
#
# This is not a defensive nicety -- it is required for scorer conformance. In the
# reference implementation an unrecognized prediction string simply never equals
# any class, so it yields no TP and no FP, but it DOES fail the "prediction hits
# some reference" guard and therefore charges a false negative to every member of
# the reference set. Encoding it as OOV_INDEX reproduces that behaviour exactly.
OOV_INDEX: Final[int] = -1


# --- Label-string normalization -------------------------------------------
#
# Replicates the official scorer's string handling. Verified behaviours:
#   * whitespace runs collapse to single spaces
#   * trailing ". ; : , ，" and whitespace are stripped
#   * matching is case-insensitive
#   * an escaped "\/" is unescaped to "/"
#   * a handful of spacing aliases are accepted
#   * anything unrecognized passes through normalized (and then never matches)

_TRAILING_PUNCT: Final[str] = " \t\r\n.;:，,"


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _strip_trailing_punct(text: str) -> str:
    return text.rstrip(_TRAILING_PUNCT)


def _norm_key(text: str) -> str:
    """Lowercased, whitespace-collapsed, trailing-punctuation-stripped key."""
    return _normalize_spaces(_strip_trailing_punct(text)).lower()


_EVASION_BY_KEY: Final[dict[str, str]] = {_norm_key(v): v for v in EVASION_LABELS}
_CLARITY_BY_KEY: Final[dict[str, str]] = {_norm_key(v): v for v in CLARITY_LABELS}

_EVASION_ALIASES: Final[dict[str, str]] = {
    "partial/half answer": "Partial/half-answer",
    "partial / half-answer": "Partial/half-answer",
    "partial / half answer": "Partial/half-answer",
}

_CLARITY_ALIASES: Final[dict[str, str]] = {
    "ambivalent reply": "Ambivalent",
    "clear nonreply": "Clear Non-Reply",
    "clear non reply": "Clear Non-Reply",
}


def normalize_evasion(label: str) -> str:
    """Canonicalize an evasion label string, or return it normalized-but-unmatched."""
    raw = str(label).replace("\\/", "/")
    key = _norm_key(raw)
    if key in _EVASION_BY_KEY:
        return _EVASION_BY_KEY[key]
    if key in _EVASION_ALIASES:
        return _EVASION_ALIASES[key]
    return _normalize_spaces(_strip_trailing_punct(raw))


def normalize_clarity(label: str) -> str:
    """Canonicalize a clarity label string, or return it normalized-but-unmatched."""
    raw = str(label)
    key = _norm_key(raw)
    if key in _CLARITY_BY_KEY:
        return _CLARITY_BY_KEY[key]
    if key in _CLARITY_ALIASES:
        return _CLARITY_ALIASES[key]
    return _normalize_spaces(_strip_trailing_punct(raw))


# --- Encoding to integer indices ------------------------------------------


def evasion_to_index(label: str) -> int:
    """Index into EVASION_LABELS, or OOV_INDEX if unrecognized."""
    canonical = normalize_evasion(label)
    try:
        return EVASION_LABELS.index(canonical)
    except ValueError:
        return OOV_INDEX


def clarity_to_index(label: str) -> int:
    """Index into CLARITY_LABELS, or OOV_INDEX if unrecognized."""
    canonical = normalize_clarity(label)
    try:
        return CLARITY_LABELS.index(canonical)
    except ValueError:
        return OOV_INDEX


def encode_evasion(labels: list[str]) -> np.ndarray:
    """Encode evasion label strings to an int array, OOV_INDEX where unrecognized."""
    return np.asarray([evasion_to_index(x) for x in labels], dtype=np.int64)


def encode_clarity(labels: list[str]) -> np.ndarray:
    """Encode clarity label strings to an int array, OOV_INDEX where unrecognized."""
    return np.asarray([clarity_to_index(x) for x in labels], dtype=np.int64)


# --- Partition application -------------------------------------------------

# Leaf index -> clarity index, under the OFFICIAL partition.
OFFICIAL_PARTITION_MAP: Final[np.ndarray] = np.asarray(
    [CLARITY_LABELS.index(OFFICIAL_CLARITY_OF_LEAF[leaf]) for leaf in EVASION_LABELS],
    dtype=np.int64,
)


def leaf_to_official_clarity(leaf_idx: np.ndarray) -> np.ndarray:
    """Map leaf indices to OFFICIAL clarity indices, preserving OOV_INDEX.

    Subtask 1 predictions are derived from Subtask 2 by this deterministic map,
    which is what both the 1st- and 2nd-place systems did.
    """
    leaf_idx = np.asarray(leaf_idx, dtype=np.int64)
    out = np.full(leaf_idx.shape, OOV_INDEX, dtype=np.int64)
    known = leaf_idx >= 0
    out[known] = OFFICIAL_PARTITION_MAP[leaf_idx[known]]
    return out


def multi_reference_mask(
    reference_labels: list[list[str]], *, n_classes: int = N_EVASION
) -> np.ndarray:
    """Build the (N, n_classes) boolean reference-set mask used by the scorer.

    `reference_labels[i]` is the list of per-annotator labels for item i, which
    may contain duplicates, empty strings or None-like entries; the mask
    deduplicates by construction, matching the scorer's use of a Python set.

    Unrecognized label strings are DROPPED from the reference set rather than
    encoded as OOV, mirroring the reference loader: it builds the gold set from
    whatever annotator fields are non-empty and normalizes each, so a string
    outside the vocabulary simply never becomes a member any prediction can hit.
    """
    n_items = len(reference_labels)
    mask = np.zeros((n_items, n_classes), dtype=bool)
    for i, refs in enumerate(reference_labels):
        for raw in refs:
            if raw is None or str(raw).strip() == "":
                continue
            idx = evasion_to_index(str(raw))
            if idx != OOV_INDEX:
                mask[i, idx] = True
    return mask
