"""Legal evidence span value object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSpan:
    """Half-open character span in an original document."""

    source_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid half-open character span: {self.start}:{self.end}")
