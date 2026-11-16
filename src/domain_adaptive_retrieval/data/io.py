"""Small, explicit data IO helpers for offline benchmark files."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records without materializing all rows."""
    with path.open(encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                yield row


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, TSV, CSV or Parquet as a list of named records."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file}")
    if file.suffix == ".jsonl":
        return list(read_jsonl(file))
    if file.suffix in {".tsv", ".csv"}:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if file.suffix == ".tsv" else ","))
    if file.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet loading requires: uv sync --extra benchmark") from exc
        return pd.read_parquet(file).to_dict("records")  # type: ignore[no-any-return]
    raise ValueError(f"Unsupported table {file}; use .jsonl, .tsv, .csv or .parquet")
