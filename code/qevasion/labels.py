"""Canonical label vocabulary for SemEval-2026 Task 6 (CLARITY)."""

from __future__ import annotations

from typing import Final

import numpy as np


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


CLARITY_LABELS: Final[tuple[str, ...]] = (
    "Ambivalent",
    "Clear Non-Reply",
    "Clear Reply",
)
N_CLARITY: Final[int] = len(CLARITY_LABELS)


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

OOV_INDEX: Final[int] = -1


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


OFFICIAL_PARTITION_MAP: Final[np.ndarray] = np.asarray(
    [CLARITY_LABELS.index(OFFICIAL_CLARITY_OF_LEAF[leaf]) for leaf in EVASION_LABELS],
    dtype=np.int64,
)


def leaf_to_official_clarity(leaf_idx: np.ndarray) -> np.ndarray:
    """Map leaf indices to OFFICIAL clarity indices, preserving OOV_INDEX."""
    leaf_idx = np.asarray(leaf_idx, dtype=np.int64)
    out = np.full(leaf_idx.shape, OOV_INDEX, dtype=np.int64)
    known = leaf_idx >= 0
    out[known] = OFFICIAL_PARTITION_MAP[leaf_idx[known]]
    return out


def multi_reference_mask(
    reference_labels: list[list[str]], *, n_classes: int = N_EVASION
) -> np.ndarray:
    """Build the (N, n_classes) boolean reference-set mask used by the scorer."""
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
