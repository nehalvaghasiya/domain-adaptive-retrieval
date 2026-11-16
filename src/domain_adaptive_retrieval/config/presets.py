"""Locate packaged example configuration files in a checkout."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def domain_config(domain: str) -> Path:
    """Find a predefined domain configuration within a source checkout."""
    path = REPO_ROOT / "configs" / "domains" / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown domain or missing config: {path}")
    return path
