"""Character-offset-preserving legal word chunker."""

import re
from collections.abc import Iterator


def chunk_words(
    text: str,
    width: int = 180,
    overlap: int = 40,
) -> Iterator[tuple[int, int, int, str]]:
    """Yield (chunk index, begin, end, exact substring) with half-open offsets."""
    if width <= 0 or not 0 <= overlap < width:
        raise ValueError("Require 0 <= overlap < chunk width")

    words = list(re.finditer(r"\S+", text))

    for index, first in enumerate(range(0, len(words), width - overlap)):
        last = min(first + width, len(words))

        start = words[first].start()
        end = words[last - 1].end()

        yield index, start, end, text[start:end]

        if last == len(words):
            break
