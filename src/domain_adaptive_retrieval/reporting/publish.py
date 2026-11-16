"""Explicit opt-in, private-by-default Hugging Face checkpoint publishing."""

from pathlib import Path
from typing import Any


def publish_checkpoint(
    checkpoint: str | Path,
    *,
    repo_id: str,
    private: bool = True,
    confirm: bool = False,
    token: str | None = None,
) -> Any:
    """Upload a local checkpoint only after explicit confirmation and license review."""
    if not confirm:
        raise PermissionError(
            "Publishing is disabled. Pass --confirm only after reviewing licenses/data privacy"
        )
    directory = Path(checkpoint)
    if not directory.is_dir() or not (directory / "README.md").is_file():
        raise FileNotFoundError(f"Checkpoint and model card required at {directory}")
    if "/" not in repo_id or repo_id.startswith("/"):
        raise ValueError("repo-id must be NAMESPACE/MODEL_NAME")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Publishing requires: uv sync --extra publish") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id, folder_path=str(directory), repo_type="model"
    )
