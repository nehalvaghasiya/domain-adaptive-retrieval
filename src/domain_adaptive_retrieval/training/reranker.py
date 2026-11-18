"""One pointwise binary cross-encoder fine-tuning service."""

import gc
import random
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
from domain_adaptive_retrieval.retrieval.candidates import (
    load_candidates,
    save_candidates,
)
from domain_adaptive_retrieval.retrieval.dense import DenseRetriever
from domain_adaptive_retrieval.retrieval.rerank import rerank_candidates
from domain_adaptive_retrieval.training.examples import reranker_pairs
from domain_adaptive_retrieval.training.progress import TrainingProgress
from domain_adaptive_retrieval.training.resume import (
    latest_resume_checkpoint,
    verified_resume_checkpoint,
)
from domain_adaptive_retrieval.training.selection import (
    checkpoints,
    choose_best_checkpoint,
)


class RerankerTrainer:
    """Train pointwise BCE and select using frozen base-retriever val candidates."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def train(self, bundle: RetrievalBundle, output: str | Path) -> Path:
        """Train an actual cross-encoder and select by validation nDCG@10."""
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Reranker training requires: uv sync --extra train"
            ) from exc
        cfg = self.config
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(cfg, bundle, target, "reranker")
        if progress.selected:
            return target / "model"
        check_device(cfg.device, require_gpu=cfg.require_gpu)
        val_qids = bundle.split_queries("val")[: cfg.max_val_queries or None]
        if not val_qids:
            raise ValueError("Reranker checkpoint selection requires validation labels")
        candidate_path = target / "validation_candidates.json"
        if candidate_path.exists():
            fixed_candidates = load_candidates(candidate_path)
        else:
            first_stage = SentenceTransformer(cfg.embedding_model, device=cfg.device)
            first_stage.max_seq_length = cfg.max_sequence_length
            fixed_candidates = DenseRetriever(
                first_stage,
                device=cfg.device,
                query_prefix=cfg.query_prefix,
                batch_size=cfg.embedding_batch_size,
                query_batch=cfg.similarity_query_batch_size,
                doc_chunk=cfg.similarity_doc_chunk_size,
            ).search(bundle, val_qids, split="val", top_k=cfg.candidate_top_k)
            save_candidates(candidate_path, fixed_candidates)
            del first_stage
            self._release_cuda()
        if not progress.optimized:
            progress.finish_optimization(self._optimize(bundle, target))
        candidates = checkpoints(target / "trainer") + [target / "last_checkpoint"]
        scores = progress.scores
        for path in candidates:
            if str(path) in scores:
                continue
            checkpoint: Any = CrossEncoder(str(path), num_labels=1, device=cfg.device)
            ranked = rerank_candidates(
                bundle, fixed_candidates, checkpoint, batch_size=cfg.reranker_batch_size
            )
            scores[str(path)], _ = evaluate_predictions(
                bundle, ranked, split="val", query_ids=val_qids
            )
            progress.record_score(path, scores[str(path)])
            del checkpoint
            self._release_cuda()
        selected = choose_best_checkpoint(scores)
        write_json(target / "validation_checkpoint_scores.json", scores)
        chosen = CrossEncoder(selected, num_labels=1, device=cfg.device)
        chosen.save_pretrained(str(target / "model"))
        write_json(
            target / "training_stats.json",
            {**progress.stats, "selected_checkpoint": selected},
        )
        progress.finish_selection(selected)
        return target / "model"

    def _optimize(self, bundle: RetrievalBundle, target: Path) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder import (
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
        )
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from transformers import set_seed

        from domain_adaptive_retrieval.training.callbacks import (
            CompleteCheckpointCallback,
        )

        cfg = self.config
        set_seed(cfg.seed)
        threshold = float(bundle.meta.get("exact_threshold", cfg.positive_threshold))
        pairs = positive_pairs(
            bundle, threshold=threshold, limit=cfg.max_train_pairs, seed=cfg.seed
        )
        mined = mine_training_negatives(
            bundle,
            pairs,
            per_positive=cfg.negatives_per_positive,
            strategy=cfg.negative_strategy,
            threshold=threshold,
            seed=cfg.seed,
        )
        rows = reranker_pairs(bundle, mined)
        random.Random(cfg.seed).shuffle(rows)
        model = CrossEncoder(
            cfg.reranker_model,
            num_labels=1,
            max_length=cfg.max_sequence_length,
            device=cfg.device,
        )
        verify_model_device(model, cfg.device)
        args = CrossEncoderTrainingArguments(
            output_dir=str(target / "trainer"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.accumulation,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            fp16=cfg.precision == "fp16",
            bf16=cfg.precision == "bf16",
            gradient_checkpointing=cfg.gradient_checkpointing,
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
        positives = sum(record["label"] == 1 for record in rows)
        negatives = len(rows) - positives
        loss = BinaryCrossEntropyLoss(
            model,
            pos_weight=torch.tensor(negatives / max(positives, 1), device=cfg.device),
        )
        trainer = CrossEncoderTrainer(
            model=model, args=args, train_dataset=Dataset.from_list(rows), loss=loss
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
        trainer.model.save_pretrained(str(target / "last_checkpoint"))
        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        )
        del trainer, model, loss
        self._release_cuda()
        return {
            "objective": "BinaryCrossEntropyLoss",
            "training_seconds": seconds,
            "training_pairs": len(rows),
            "positives": positives,
            "negatives": negatives,
            "peak_cuda_memory_bytes": peak,
            "validation_metric": "ndcg@10",
            "candidate_source": cfg.embedding_model,
            "resumed_from": resume,
            "weak_negative_warning": True,
        }

    @staticmethod
    def _release_cuda() -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
