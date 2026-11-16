"""Validated experiment settings shared by CLI and notebooks."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    """Single experiment configuration with explicit, documented defaults."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    domain: Literal[
        "finance", "biomedical", "legal", "ecommerce", "code", "custom"
    ] = "finance"
    device: str = "cpu"
    require_gpu: bool = False
    precision: Literal["fp16", "fp32", "bf16"] = "fp32"
    seed: int = 42
    cache_dir: Path = Path("data/cache")
    custom_dir: Path | None = None
    legal_dir: Path | None = None
    esci_dir: Path | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_sequence_length: int = Field(default=256, ge=16)
    epochs: float = Field(default=2.0, gt=0)
    batch_size: int = Field(default=8, ge=1)
    accumulation: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    gradient_checkpointing: bool = False
    resume_checkpoint: Path | None = None
    max_train_pairs: int = Field(default=0, ge=0)
    max_val_queries: int = Field(default=100, ge=0)
    max_test_queries: int = Field(default=0, ge=0)
    negative_strategy: Literal["bm25", "random"] = "bm25"
    negatives_per_positive: int = Field(default=2, ge=1)
    positive_threshold: float = 1.0
    embedding_batch_size: int = Field(default=32, ge=1)
    reranker_batch_size: int = Field(default=16, ge=1)
    similarity_query_batch_size: int = Field(default=16, ge=1)
    similarity_doc_chunk_size: int = Field(default=8192, ge=1)
    candidate_top_k: int = Field(default=100, ge=1)
    legal_chunk_words: int = Field(default=180, ge=1)
    legal_chunk_overlap: int = Field(default=40, ge=0)
    esci_locale: str = "us"
    code_max_train: int = Field(default=15000, ge=0)
    code_max_val: int = Field(default=500, ge=0)
    code_max_test: int = Field(default=500, ge=0)
    code_max_chars: int = Field(default=4000, ge=100)
    code_max_query_chars: int = Field(default=500, ge=20)
    publish: bool = False
    hub_private: bool = True

    @field_validator("device", mode="before")
    @classmethod
    def normalize_runtime_device(cls, value: str) -> str:
        from domain_adaptive_retrieval.models.device import normalize_device

        return normalize_device(value)

    @model_validator(mode="after")
    def validate_runtime(self) -> "ExperimentConfig":
        if self.require_gpu and not self.device.startswith("cuda:"):
            raise ValueError(
                "require_gpu=true requires device=cuda:0 (T4) or another CUDA index"
            )
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU runs require precision=fp32")
        if self.legal_chunk_overlap >= self.legal_chunk_words:
            raise ValueError("legal_chunk_overlap must be less than legal_chunk_words")
        return self
