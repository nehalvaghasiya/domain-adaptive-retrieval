"""Typed entry point for loading a Sentence Transformers bi-encoder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.models.device import (
    check_device,
    normalize_device,
    verify_model_device,
)


class EmbeddingModel:
    """Load on the requested device and forward the encoder's batching options."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding models require: uv sync --extra train"
            ) from exc

        self.device = normalize_device(device)
        check_device(self.device)
        self.model = SentenceTransformer(str(checkpoint), device=self.device)
        self.model.max_seq_length = max_length
        verify_model_device(self.model, self.device)

    def encode(self, texts: Sequence[str], **options: Any) -> Any:
        """Encode query or document texts using caller-supplied inference options."""
        return self.model.encode(list(texts), **options)
