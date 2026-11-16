"""Cross-split deduplication for code and exact text."""

import ast
import textwrap

from domain_adaptive_retrieval.data.cleaning import clean_text, normalized_sha


def remove_python_docstring(code: str) -> str:
    """Remove AST-recognized Python function/class docstrings, not all strings."""
    source = textwrap.dedent(clean_text(code))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source
    deleted: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                deleted.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return "".join(line for line_no, line in enumerate(source.splitlines(keepends=True), 1) if line_no not in deleted).strip()


def remove_duplicate_texts(corpus: dict[str, str]) -> dict[str, str]:
    """Keep first document by sorted ID for identical whitespace-normalized text."""
    seen: set[str] = set()
    kept: dict[str, str] = {}
    for did in sorted(corpus):
        sha = normalized_sha(corpus[did])
        if sha not in seen:
            kept[did] = corpus[did]
            seen.add(sha)
    return kept
