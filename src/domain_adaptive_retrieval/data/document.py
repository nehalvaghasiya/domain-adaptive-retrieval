"""Document value object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One indexed document, with optional original source ID."""

    id: str
    text: str
    source_id: str | None = None
