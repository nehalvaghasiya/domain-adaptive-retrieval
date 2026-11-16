"""Text cleaning at dataset boundaries, never after splitting qrels."""

import hashlib
import re


def clean_text(value: object) -> str:
    """Normalize nullable text while preserving internal character offsets."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def stable_id(*parts: object) -> str:
    """Deterministic, compact ID for constructed retrieval examples."""
    return hashlib.sha256("\u241e".join(str(p) for p in parts).encode()).hexdigest()[:20]


def normalized_sha(text: str) -> str:
    """SHA-256 of whitespace-normalized content for duplicate detection."""
    return hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()
