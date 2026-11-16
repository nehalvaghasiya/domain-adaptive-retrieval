"""Docstrings must not be included in indexed source and duplicates rejected."""

from domain_adaptive_retrieval.data.cleaning import normalized_sha
from domain_adaptive_retrieval.data.dedupe import (
    remove_duplicate_texts,
    remove_python_docstring,
)


def test_python_function_docstring_removed() -> None:
    code = (
        'def foo(x):\n    """Explain foo secret retrieval label"""\n    return x * 2\n'
    )
    cleaned = remove_python_docstring(code)
    assert "secret retrieval label" not in cleaned
    assert "return x * 2" in cleaned


def test_duplicate_normalized_code() -> None:
    assert normalized_sha("a b") == normalized_sha("ab")
    assert len(remove_duplicate_texts({"one": "a b", "two": "ab", "three": "xyz"})) == 2
