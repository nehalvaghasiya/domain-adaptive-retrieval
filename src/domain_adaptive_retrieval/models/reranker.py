"""Typed entry point for loading a Sentence Transformers cross-encoder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.models.device import (
    check_device,
    normalize_device,
    verify_model_device,
)


class RerankerModel:
    """Load a single-score cross-encoder and verify its parameter placement."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranker models require: uv sync --extra train"
            ) from exc

        self.device = normalize_device(device)
        check_device(self.device)
        self.model = CrossEncoder(
            str(checkpoint), device=self.device, max_length=max_length, num_labels=1
        )
        verify_model_device(self.model, self.device)

    def predict(self, pairs: Sequence[tuple[str, str]], **options: Any) -> Any:
        """Score query/document pairs with caller-supplied inference options."""
        return self.model.predict(list(pairs), **options)
