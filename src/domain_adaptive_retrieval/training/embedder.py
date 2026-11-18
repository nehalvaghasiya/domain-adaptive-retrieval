"""One bi-encoder training service using Sentence Transformers MNRL."""

import gc
import time
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.evaluation.evaluator import evaluate_predictions
from domain_adaptive_retrieval.mining.bm25_miner import mine_training_negatives
from domain_adaptive_retrieval.mining.sampler import positive_pairs
from domain_adaptive_retrieval.models.device import check_device, verify_model_device
from domain_adaptive_retrieval.reporting.artifacts import write_json
from domain_adaptive_retrieval.retrieval.dense import DenseRetriever
from domain_adaptive_retrieval.training.examples import embedding_triplets
from domain_adaptive_retrieval.training.progress import TrainingProgress
from domain_adaptive_retrieval.training.resume import (
    latest_resume_checkpoint,
    verified_resume_checkpoint,
)
from domain_adaptive_retrieval.training.selection import (
    checkpoints,
    choose_best_checkpoint,
)


class EmbeddingTrainer:
    """Train and select BGE embedder checkpoints on validation ranking quality."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def train(self, bundle: RetrievalBundle, output: str | Path) -> Path:
        """Perform actual optimizer steps, save/reload candidates, select best on val."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Training requires: uv sync --extra train") from exc
        cfg = self.config
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(cfg, bundle, target, "embedding")
        if progress.selected:
            return target / "model"
        check_device(cfg.device, require_gpu=cfg.require_gpu)
        if not progress.optimized:
            progress.finish_optimization(self._optimize(bundle, target))
        stats = progress.stats
        candidates = checkpoints(target / "trainer") + [target / "last_checkpoint"]
        scores = progress.scores
        qids = bundle.split_queries("val")[: cfg.max_val_queries or None]
        if not qids:
            raise ValueError(
                "Embedding checkpoint selection requires validation labels"
            )
        for path in candidates:
            if str(path) in scores:
                continue
            model = SentenceTransformer(str(path), device=cfg.device)
            model.max_seq_length = cfg.max_sequence_length
            rankings = DenseRetriever(
                model,
                device=cfg.device,
                query_prefix=cfg.query_prefix,
                batch_size=cfg.embedding_batch_size,
                query_batch=cfg.similarity_query_batch_size,
                doc_chunk=cfg.similarity_doc_chunk_size,
            ).search(bundle, qids, split="val", top_k=cfg.candidate_top_k)
            scores[str(path)], _ = evaluate_predictions(
                bundle, rankings, split="val", query_ids=qids
            )
            progress.record_score(path, scores[str(path)])
            del model
            self._release_cuda()
        selected = choose_best_checkpoint(scores)
        write_json(target / "validation_checkpoint_scores.json", scores)
        best = SentenceTransformer(selected, device=cfg.device)
        best.save(str(target / "model"))
        write_json(
            target / "training_stats.json", {**stats, "selected_checkpoint": selected}
        )
        progress.finish_selection(selected)
        return target / "model"

    def _optimize(self, bundle: RetrievalBundle, target: Path) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from transformers import set_seed

        from domain_adaptive_retrieval.training.callbacks import (
            CompleteCheckpointCallback,
        )

        cfg = self.config
        set_seed(cfg.seed)
        positive = positive_pairs(
            bundle,
            threshold=self._threshold(bundle),
            limit=cfg.max_train_pairs,
            seed=cfg.seed,
        )
        mined = mine_training_negatives(
            bundle,
            positive,
            per_positive=cfg.negatives_per_positive,
            strategy=cfg.negative_strategy,
            threshold=self._threshold(bundle),
            seed=cfg.seed,
        )
        rows = embedding_triplets(bundle, mined, query_prefix=cfg.query_prefix)
        if not rows:
            raise ValueError("No suitable embedding training triplets")
        model = SentenceTransformer(cfg.embedding_model, device=cfg.device)
        model.max_seq_length = cfg.max_sequence_length
        verify_model_device(model, cfg.device)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(target / "trainer"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.accumulation,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            gradient_checkpointing=cfg.gradient_checkpointing,
            fp16=cfg.precision == "fp16",
            bf16=cfg.precision == "bf16",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="steps" if cfg.save_steps else "epoch",
            save_steps=cfg.save_steps or 500,
            save_only_model=False,
            save_total_limit=max(2, int(cfg.epochs) + 1),
            eval_strategy="no",
            logging_steps=max(1, len(rows) // (cfg.batch_size * 10)),
            report_to="none",
            dataloader_num_workers=0,
            seed=cfg.seed,
            use_cpu=cfg.device == "cpu",
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(rows),
            loss=MultipleNegativesRankingLoss(model),
        )
        trainer.add_callback(CompleteCheckpointCallback())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        requested = cfg.resume_checkpoint or latest_resume_checkpoint(
            target / "trainer"
        )
        resume = verified_resume_checkpoint(requested, target / "trainer")
        if resume is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=resume)
        seconds = time.perf_counter() - start
        trainer.model.save(str(target / "last_checkpoint"))
        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        )
        del model, trainer
        self._release_cuda()
        return {
            "objective": "MultipleNegativesRankingLoss",
            "training_seconds": seconds,
            "training_pairs": len(rows),
            "peak_cuda_memory_bytes": peak,
            "validation_metric": "ndcg@10",
            "resumed_from": resume,
            "weak_negative_warning": True,
        }

    @staticmethod
    def _release_cuda() -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _threshold(self, bundle: RetrievalBundle) -> float:
        return float(bundle.meta.get("exact_threshold", self.config.positive_threshold))
