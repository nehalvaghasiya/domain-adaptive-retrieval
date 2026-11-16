"""Stable fingerprints and sizes for local checkpoint files."""

import hashlib
from pathlib import Path


def _checkpoint_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def fingerprint_path(path: str | Path) -> str:
    """Hash relative filenames and streamed contents without loading model weights."""
    root = Path(path)
    files = _checkpoint_files(root)
    base = root if root.is_dir() else root.parent
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_size_mb(path: str | Path) -> float:
    """Return the checkpoint's on-disk size in MiB."""
    return sum(file.stat().st_size for file in _checkpoint_files(Path(path))) / 1024**2
